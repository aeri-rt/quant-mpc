"""scripts/calibrate_costs.py: calibrate() on hand-built frames, main() exit 2 without broker
fills, and the Runner's one-shot cost-model WARNING on the first broker STK fill."""

import pandas as pd
import pytest
from loguru import logger

from scripts.calibrate_costs import TOL, _off, calibrate, main
from src.config import Costs
from src.costs import commission, reg_fees
from src.store import Store
from src.types import Fill, Order

STOCK = 756733
NO_BAGS = pd.DataFrame(columns=["session_id", "id"])
approx = pytest.approx


def frame(rows, source="broker", con_id=STOCK, order_id=None):
    """rows: (qty, price, commission); one order per fill unless `order_id` is given."""
    return pd.DataFrame(
        [
            {
                "exec_id": f"e{i}",
                "session_id": "s1",
                "order_id": order_id or f"o{i}",
                "con_id": con_id,
                "qty": q,
                "price": p,
                "commission": cm,
                "source": source,
            }
            for i, (q, p, cm) in enumerate(rows)
        ]
    )


@pytest.fixture
def c():
    return Costs()  # Tiered defaults: 0.0035/sh, 0.35 min


def test_tiered_fills_match_the_model(c):
    m = c.min_commission
    sell = commission(-100, c) + reg_fees(-100, 600.0, c)
    f = frame(
        [(4, 600.0, m), (100, 600.0, m), (200, 600.0, commission(200, c))]
        + [(-100, 600.0, sell)] * 2
    )
    r = calibrate(f, NO_BAGS, c)
    assert r["n_broker_fills"] == 5 and r["n_stk"] == 5 and r["n_null_commission"] == 0
    assert r["min_binds_frac"] == approx(2 / 3)
    assert r["effective_buy_rate"] == approx((m + m + commission(200, c)) / (4 + 100 + 200))
    assert r["unbound_buy_rate"] == approx(c.commission_per_share)
    assert r["inferred_sec_rate"] == approx(c.sec_fee_rate, rel=0.05)
    assert r["stk_ratio"] == approx(1.0) and r["mean_rel_err"] == approx(0.0)
    assert r["yaml"] == []


def test_fixed_pricing_suggests_min_and_rate(c):
    r = calibrate(frame([(4, 600.0, 1.0), (100, 600.0, 1.0), (200, 600.0, 1.0)]), NO_BAGS, c)
    assert r["min_binds_frac"] == 0
    y = "\n".join(r["yaml"])
    assert "min_commission: 1.0" in y and "commission_per_share" in y


def test_large_fills_only_do_not_invent_a_minimum(c):
    r = calibrate(frame([(200, 600.0, commission(200, c))] * 3), NO_BAGS, c)
    assert r["min_binds_frac"] == 0 and r["yaml"] == []


def test_minimum_applies_per_order_not_per_execution(c):
    """A 100-share hedge filled 60 + 40: IBKR books ONE $0.35 minimum, split over the reports."""
    r = calibrate(frame([(60, 600.0, 0.21), (40, 600.0, 0.14)], order_id="o1"), NO_BAGS, c)
    assert r["n_stk"] == 1 and r["n_stk_fills"] == 2
    assert r["stk_modeled"] == approx(c.min_commission) and r["stk_ratio"] == approx(1.0)
    assert r["min_binds_frac"] == 1.0 and r["yaml"] == []
    # an order with one report still missing is not compared at all
    r = calibrate(frame([(60, 600.0, 0.21), (40, 600.0, None)], order_id="o1"), NO_BAGS, c)
    assert r["n_null_commission"] == 1 and "n_stk" not in r


def test_off_has_an_absolute_floor():
    assert not _off(1e-9, 0.0) and _off(2.78e-5, 0.0)  # a zero config must not flag noise
    assert _off(0.5, 0.35, TOL) and not _off(0.352, 0.35, TOL)


def test_bag_orders_against_option_model(c):
    k = 10  # 5 straddles = 2 legs x 5 contracts
    per_leg = (
        max(c.option_min_commission, c.option_commission * k) + c.option_fees_per_contract * k
    ) / 2
    f = frame([(5, 3.0, per_leg), (5, 3.1, per_leg)], con_id=1001, order_id="bag1")
    f.loc[1, "con_id"] = 1002
    bags = pd.DataFrame({"session_id": ["s1"], "id": ["bag1"]})
    r = calibrate(f, bags, c)
    assert "n_stk" not in r and r["n_bag_orders"] == 1 and r["opt_ratio"] == approx(1.0)


def test_sim_fills_are_ignored(c):
    assert calibrate(frame([(100, 600.0, 0.35)], source="sim"), NO_BAGS, c) == {"n_broker_fills": 0}
    nulls = frame([(100, 600.0, None)])
    assert calibrate(nulls, NO_BAGS, c)["n_null_commission"] == 1


def test_main_exit_2_without_broker_fills(tmp_path, sim_clock):
    db = tmp_path / "t.db"
    s = Store(db)
    s.start_session("paper", "{}", sim_clock.now())
    s.close()
    assert main(["--db", str(db)]) == 2
    assert main(["--db", str(tmp_path / "missing.db")]) == 2


async def test_runner_warns_once_on_cost_model_mismatch(runner, fake_market, sim_clock):
    r = runner(fake_market)
    await r.start()
    logged = []
    sink = logger.add(logged.append, level="WARNING")

    def booked(exec_id, order_id, qty, order_qty, comm):
        """A broker fill of `qty` on an order of `order_qty` with its commission report."""
        r.broker.orders[order_id] = Order(
            order_id, STOCK, "STK", "BUY", order_qty, 600.01, "hedge", created=sim_clock.now()
        )
        fill = Fill(exec_id, order_id, STOCK, qty, 600.0, sim_clock.now(), None, "broker")
        r.broker.fills.append(fill)
        r.broker.commission_updates.append((exec_id, comm))

    try:
        booked("p0", "p0", 40, 100, 0.14)  # one execution of a 100-share order: never compared
        sim_clock.advance(60)
        await r.tick()
        assert not r._cost_checked and not any("cost model mismatch" in m for m in logged)
        for i, comm in enumerate((0.70, 0.90)):  # modeled 0.35 for 100 shares: both mismatch
            booked(f"x{i}", f"o{i}", 100, 100, comm)
            sim_clock.advance(60)
            await r.tick()
    finally:
        logger.remove(sink)
    warnings = [m for m in logged if "cost model mismatch" in m]
    assert r._cost_checked and len(warnings) == 1
    assert "actual=0.70 modeled=0.35" in warnings[0] and "calibrate_costs" in warnings[0]


def test_main_reads_fills_and_bag_orders_from_the_store(tmp_path, sim_clock, capsys):
    db = tmp_path / "t.db"
    s = Store(db)
    now = sim_clock.now()
    old = s.start_session("paper", "{}", now)  # excluded by --last 1
    sim_clock.advance(60)
    sid = s.start_session("paper", "{}", sim_clock.now())
    for session, oid in ((old, "stk0"), (sid, "stk1")):
        s.save_order(session, Order(oid, STOCK, "STK", "BUY", 100, 600.01, "hedge", created=now))
        s.save_fill(session, Fill(f"{oid}-x", oid, STOCK, 100, 600.0, now, 0.35, "broker"))
    s.save_order(sid, Order("bag1", 0, "BAG", "BUY", 5, 30.0, "entry", created=now))
    for i, cid in enumerate((1001, 1002)):
        s.save_fill(sid, Fill(f"bag-{i}", "bag1", cid, 5, 15.0, now, 3.5, "broker"))
    s.save_fill(sid, Fill("sim-1", "stk1", STOCK, -100, 600.0, now, 0.35, "sim"))
    s.close()
    assert main(["--db", str(db), "--last", "1"]) == 0
    out = capsys.readouterr().out
    assert "skipping 1 sim fills" in out and "STK 1 orders (1 fills)" in out
    assert "OPT 1 BAG orders" in out and "ratio 1.000" in out and "nothing to change" in out
    assert main(["--db", str(db), "--session", old]) == 0
    assert "STK 1 orders (1 fills)" in capsys.readouterr().out
