"""Compare what IBKR actually charged with the cost model (src.costs) over stored broker fills.

The band width, the QP side cost and the gate all take `costs` from config/live.yaml. This reads
data/trades.db (never TWS), prints actual vs modelled commissions per order class (STK orders,
BAG orders; IBKR's minimums are per order, not per execution) and the YAML lines to change when
they differ by more than 20%. Sim fills are skipped: the model charged them.

usage: calibrate_costs.py [--db data/trades.db] [--session ID | --last N] [--config config/live.yaml]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Costs, load_live
from src.costs import commission, reg_fees
from src.store import Store

TOL = 0.005  # $: a fill "at the minimum", or "bound by it"
DRIFT = 0.2  # relative difference from the config that earns a YAML line


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", default="data/trades.db")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--session", help="one session id (default: every session)")
    g.add_argument("--last", type=int, metavar="N", help="the N most recent sessions")
    p.add_argument("--config", default="config/live.yaml")
    return p.parse_args(argv)


def load(store: Store, session: str | None, last: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fills and BAG orders (session_id, id) of the selected sessions."""
    cond, params = "1=1", []
    if session:
        cond, params = "session_id = ?", [session]
    elif last:
        cond = "session_id IN (SELECT id FROM sessions ORDER BY started_at DESC LIMIT ?)"
        params = [last]
    fills = pd.read_sql_query(
        "SELECT exec_id, session_id, order_id, con_id, qty, price, commission, source"
        f" FROM fills WHERE {cond}",
        store._db,
        params=params,
    )
    bags = pd.read_sql_query(
        f"SELECT session_id, id FROM orders WHERE sec_type = 'BAG' AND {cond}",
        store._db,
        params=params,
    )
    return fills, bags


def _off(actual: float, config: float, atol: float = 1e-6) -> bool:
    """Relative drift > DRIFT, with an absolute floor so a zero config does not flag noise."""
    return abs(actual - config) > max(DRIFT * abs(config), atol)


def stk_orders(stk: pd.DataFrame) -> pd.DataFrame:
    """One row per STK order: signed qty, VWAP price, booked commission, fill count. IBKR's $0.35
    minimum and the TAF cap apply per ORDER, so a hedge filled in several executions is compared
    at order level; an order with any unreported commission is dropped."""
    g = stk.assign(notional=stk.price * stk.qty.abs()).groupby(["session_id", "order_id"])
    o = g.agg(
        qty=("qty", "sum"),
        notional=("notional", "sum"),
        commission=("commission", lambda s: s.sum(skipna=False)),
        fills=("qty", "size"),
    ).dropna(subset=["commission"])
    return o.assign(price=o.notional / o.qty.abs()).reset_index()


def calibrate(fills: pd.DataFrame, bag_orders: pd.DataFrame, c: Costs) -> dict:
    """Actual vs modelled commissions over broker-source fills; `yaml` lists the lines to change.

    STK fills are the ones not booked under a BAG order, grouped per order (`stk_orders`). The
    per-share rate is read from buy orders the minimum does not bind (an order at the minimum says
    nothing about the rate); the observed floor is reported as the minimum only when some order at
    the floor is above rate * shares. The SEC rate comes from sells net of the commission-only
    model and the (capped) TAF, so it is compared with `sec_fee_rate` directly."""
    f = fills[fills.source == "broker"]
    if f.empty:
        return {"n_broker_fills": 0}
    is_opt = (f.session_id + "/" + f.order_id).isin(bag_orders.session_id + "/" + bag_orders.id)
    res: dict = {
        "n_broker_fills": len(f),
        "n_null_commission": int(f.commission.isna().sum()),
        "yaml": [],
    }
    stk = stk_orders(f[~is_opt])
    if len(stk):
        qty, price = stk.qty.to_numpy(int), stk.price.to_numpy(float)
        actual = stk.commission.to_numpy(float)
        modeled = np.array(
            [commission(q, c) + reg_fees(q, p, c) for q, p in zip(qty, price, strict=True)]
        )
        pos = actual > 0
        res.update(
            n_stk=len(stk),
            n_stk_fills=int(stk.fills.sum()),
            stk_actual=actual.sum(),
            stk_modeled=modeled.sum(),
            stk_ratio=actual.sum() / modeled.sum(),
            mean_rel_err=float(np.mean(np.abs(actual - modeled)[pos] / actual[pos]))
            if pos.any()
            else float("nan"),
        )
        buys, sells = qty > 0, qty < 0
        if buys.any():
            ab, qb = actual[buys], np.abs(qty[buys])
            bound = np.abs(ab - c.min_commission) <= TOL
            rate = ab[~bound].sum() / qb[~bound].sum() if (~bound).any() else None
            res.update(
                effective_buy_rate=ab.sum() / qb.sum(),
                min_binds_frac=float(bound.mean()),
                unbound_buy_rate=rate,
            )
            if rate is not None and _off(rate, c.commission_per_share):
                res["yaml"].append(f"commission_per_share: {rate:.4g}")
            floor = ab.min()
            at_floor = ab <= floor + TOL
            r = c.commission_per_share if rate is None else rate
            if _off(floor, c.min_commission, TOL) and (ab[at_floor] > r * qb[at_floor] + TOL).any():
                res["yaml"].append(f"min_commission: {floor:.2f}")
        if sells.any():
            qs, ps = np.abs(qty[sells]), price[sells]
            comm_only = sum(commission(int(q), c) for q in qs)
            taf = np.minimum(c.finra_taf_per_share * qs, c.finra_taf_cap).sum()
            sec = (actual[sells].sum() - comm_only - taf) / (ps * qs).sum()
            res["inferred_sec_rate"] = sec
            if _off(sec, c.sec_fee_rate):
                res["yaml"].append(f"sec_fee_rate: {sec:.3g}")
    opt = f[is_opt & f.commission.notna()]
    if len(opt):
        g = opt.assign(k=opt.qty.abs()).groupby(["session_id", "order_id"])
        k, act = g.k.sum().to_numpy(float), g.commission.sum().to_numpy(float)
        mod = np.maximum(c.option_min_commission, c.option_commission * k)
        mod += c.option_fees_per_contract * k
        res.update(n_bag_orders=len(k), opt_actual=act.sum(), opt_modeled=mod.sum())
        res["opt_ratio"] = act.sum() / mod.sum()
    return res


def report(r: dict, c: Costs) -> None:
    print(
        f"broker fills: {r['n_broker_fills']} "
        f"({r['n_null_commission']} without a commission report, skipped)"
    )
    if "n_stk" in r:
        print(
            f"STK {r['n_stk']} orders ({r['n_stk_fills']} fills): actual {r['stk_actual']:.2f}"
            f" modeled {r['stk_modeled']:.2f} ratio {r['stk_ratio']:.3f}"
            f" mean |actual-modeled|/actual {r['mean_rel_err']:.1%}"
        )
    if "effective_buy_rate" in r:
        rate = r["unbound_buy_rate"]
        print(
            f"  buys: effective {r['effective_buy_rate']:.5f} $/sh"
            f" (config commission_per_share {c.commission_per_share}),"
            f" minimum binds on {r['min_binds_frac']:.0%} of fills,"
            f" rate where it does not: {'n/a' if rate is None else f'{rate:.5f}'}"
        )
    if "inferred_sec_rate" in r:
        print(
            f"  sells: inferred SEC rate {r['inferred_sec_rate']:.3g}"
            f" (config sec_fee_rate {c.sec_fee_rate:.3g},"
            f" finra_taf_per_share {c.finra_taf_per_share})"
        )
    if "n_bag_orders" in r:
        print(
            f"OPT {r['n_bag_orders']} BAG orders: actual {r['opt_actual']:.2f}"
            f" modeled {r['opt_modeled']:.2f} ratio {r['opt_ratio']:.3f}"
        )
    if r["yaml"]:
        print("config/live.yaml costs: change\n  " + "\n  ".join(r["yaml"]))
    else:
        print(f"config/live.yaml costs: within {DRIFT:.0%}, nothing to change")


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    c = load_live(a.config).costs
    if not Path(a.db).exists():
        print(f"{a.db} does not exist: run an --execute session first")
        return 2
    store = Store(a.db)
    try:
        fills, bags = load(store, a.session, a.last)
    finally:
        store.close()
    if n_sim := int((fills.source == "sim").sum()):
        print(f"skipping {n_sim} sim fills (charged by the model itself)")
    r = calibrate(fills, bags, c)
    if not r["n_broker_fills"]:
        print("no broker fills: run an --execute session first")
        return 2
    report(r, c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
