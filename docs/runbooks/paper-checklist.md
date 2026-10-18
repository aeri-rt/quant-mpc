# Paper checklist — signed off before the first `--live`

Every row is pass/fail. Run the steps in order, on a regular trading day, in a **paper account
that holds nothing else in SPY** and whose NLV has been reset to the intended live size (runbook,
Prerequisites). Record the observed value in the last column; a FAIL anywhere means no `--live`.

Shell helpers used below (venv active, repo root):

```bash
DB=data/trades.db
LAST="(select id from sessions order by started_at desc limit 1)"
q() { sqlite3 -header -column "$DB" "$1"; }
LOG=$(ls -t logs/quant-mpc_*.jsonl | head -1)   # newest file: the date is the start/rotation day
```

## 0. Offline

| Check | Command | Expected (where to look) | Pass/Fail · observed |
|---|---|---|---|
| Test suite green | `pytest -q` | all passed, 0 failed, about 20 s, no TWS needed (terminal) | |
| Config loads | `python -c "from src.config import load_live; print(load_live().greeks)"` | prints the `greeks` section; a typo in `config/live.yaml` fails here, not at 09:30 | |

## 1. `python scripts/check_ibkr.py` (read-only; TWS on 7497, market open)

| Check | Command / line | Expected (where to look) | Pass/Fail · observed |
|---|---|---|---|
| Clock offset | `[ok] clock_offset_s: …` | \|offset\| < 2 s | |
| Market data type | `[ok] SPY quote: … md=…` | `md=1`; `[WARN] delayed data` must **not** print | |
| Quote freshness | same line, `age_s=` | < 5 s on a live feed (the script itself only refuses > 15 s) | |
| Chain | `[ok] chain: N quotes, M two-sided strikes, expiry YYYY-MM-DD` | N > 0, M ≥ 1; expiry is a 3rd Friday 25–35 days out (`position.entry_dte` 30 ± 5, `position.prefer_monthly: true`) | |
| Leg quote | `[ok] leg_quote: bid=… ask=… iv=…` | two-sided, `iv` not `None` | |
| Vega ratio | `[ok] greeks units: ratio_vega=…` | in [0.7, 1.4] → flag `true`; in [70, 140] → flag `false`; anything else is a FAIL to investigate, not to guess | ratio_vega = |
| Theta ratio | same line, `ratio_theta=` | in [0.7, 1.4] → `true`; in [255, 511] → `false` | ratio_theta = |
| Flags set | the two printed values set under `greeks:` in `config/live.yaml` | `grep ibkr_ config/live.yaml` shows them | vega: theta: |
| BAG whatIf | `[ok] whatIf 1 straddle: initMarginChange=…` | a number (the paper simulator accepts a SMART BAG) | |
| Account | `[ok] account: NLV=… DayTradesRemaining=…` | NLV = the intended live size, not 1,000,000; no `[WARN] NLV … > 5x` line | NLV = DTR = |
| Size | `max contracts by risk fractions: n (why, premium per straddle P)` | n ≥ 1 (≥ 5 at the intended live NLV), why `ok` (`min_nlv` / `premium_frac` mean the NLV is too small) | n = P = |
| Summary | last line, exit code | `SUMMARY: all checks passed`, `echo $?` → 0 | |

## 2. Dry-run, 30 minutes (`python scripts/run_paper.py`, started after 10:00 ET)

Nothing reaches TWS: orders go to the SimBroker, `fill_source = sim` on every row.

| Check | Command | Expected (where to look) | Pass/Fail · observed |
|---|---|---|---|
| Banner | first console line | `run: mode=dry_run port=7497 client_id=10 … force_open=False` | |
| Started | log | `started dry_run: 0 straddles, 0 shares, vol sessions 29` (30 only when started after the close) then `STARTING --started--> RUNNING` (vol sessions ≥ 5; if the warm-up line says `exceeded 120.0 s`, restart in quiet hours) | |
| One row per tick | `q "select count(*), min(ts), max(ts) from decisions where session_id = $LAST"` | count ≈ minutes elapsed (`loop.interval_s` 60) | |
| Live data on every row | `q "select md_type, count(*) from decisions where session_id = $LAST group by md_type"` | only `md_type = 1` | |
| Fresh quotes | `q "select max(quote_age_s), max(clock_offset_s) from decisions where session_id = $LAST"` | age < 15 (no DEGRADED), offset < 2 | |
| Gate inputs populated | `q "select ts, S, iv_atm, sigma_f, round(edge_day,1), gate_state, reason from decisions where session_id = $LAST order by id desc limit 5"` | `iv_atm` and `sigma_f` non-NULL, `edge_day` a number (negative is the normal reading), `gate_state` CLOSED, `reason` NULL (no entry attempted) | edge_day = |
| Flat book → no hedger columns | `q "select count(*) from decisions where session_id = $LAST and band_u is not null"` | 0 (the hedge step is skipped on a flat book; NULL, not zero) | |
| Clean stop | Ctrl-C, then `q "select status, final_shares from sessions where id = $LAST"`; `echo $?` | log `stop requested: signal` … `STOPPING --stopped--> STOPPED`; status `stopped`; exit 0 | |

### 2b. Dry-run with a forced entry, 10 minutes (`python scripts/run_paper.py --force-open`)

| Check | Command | Expected (where to look) | Pass/Fail · observed |
|---|---|---|---|
| Simulated straddle | `q "select sec_type, action, qty, reason, state, avg_price from orders where session_id = $LAST"` | one row `BAG BUY n entry FILLED` (n = `max contracts` from step 1, capped at 10) | |
| Simulated fills | `q "select con_id, qty, price, commission, source from fills where session_id = $LAST"` | 2 rows, distinct `con_id`, `source = sim`, commission > 0 | |
| Hedger acts | `q "select ts, shares, delta_net, band_h, band_u, qp_u, naive_u, u_sent, risk_result, reason from decisions where session_id = $LAST and band_u is not null order by id desc limit 10"` | columns populated; `force_open = 1` on every row (`q "select min(force_open) from decisions where session_id = $LAST"` → 1) | |
| Trades only on crossings | `q "select count(*) from decisions where session_id = $LAST and band_u != 0 and abs(delta_net) <= band_h and reason != 'close_hedge'"` | 0 | |
| Stop | Ctrl-C | status `stopped`; TWS Portfolio shows **no** SPY position (nothing was real) | |

## 3. Execute, 1 hour (`python scripts/run_paper.py --execute --contracts 1 --force-open`)

Start before 15:00 ET (no entry inside the last 30 min, `loop.no_entry_last_s`), not the session
before a date in `instrument.ex_dividend_dates`, on live data (`--execute` refuses to enter on
`md_type` ≠ 1: rows show `reason = delayed`). Real paper orders from here on.

| Check | Command | Expected (where to look) | Pass/Fail · observed |
|---|---|---|---|
| Banner | first console line | `run: mode=paper … contracts=1 force_open=True` | |
| Straddle fills as one BAG | `q "select id, sec_type, action, qty, limit_price, reason, state, filled, avg_price from orders where session_id = $LAST"` | exactly one `BAG BUY 1 entry FILLED` row, no single-leg OPT order; log `entered 1 straddles K=… @ … (FILLED)` | avg_price = |
| Two leg fills | `q "select con_id, qty, price, commission, source from fills where session_id = $LAST"` | 2 rows, distinct `con_id`, `qty = 1`, `source = broker` | |
| TWS agrees | TWS Portfolio | 1 SPY call + 1 SPY put, same strike/expiry; Trade Log shows the combo | |
| Hedges only on band crossings | `q "select sum(u_sent is not null) sent, sum(reason='hedge') hedge_rows, sum(reason='in_band') in_band from decisions where session_id = $LAST"` then `q "select count(*) from decisions where session_id = $LAST and u_sent is not null and abs(delta_net) <= band_h and reason != 'close_hedge'"` | second query 0; `sent ≤ hedge_rows`; with 1 contract `band_h` ≈ 15–25 shares at 30 DTE (≈ 20–35 near the roll), so expect 0–3 hedges in an hour (0 is a pass if no row has \|delta_net\| > band_h) | sent = |
| Hedge orders are marketable limits | `q "select action, qty, limit_price, reason, state, filled from orders where session_id = $LAST and sec_type = 'STK'"` | every row `FILLED` (or `CANCELLED` after 20 s with a `working_order` row following), `limit_price` = ask + 0.01 / bid − 0.01 at the decision quote, `reason` hedge/close_hedge | |
| Commissions arrive | `q "select exec_id, con_id, qty, commission from fills where session_id = $LAST"` | every `commission` non-NULL and > 0 within a few seconds of the fill (commissionReport patched by execId) | |
| Risk never blocked a reducing trade | `q "select risk_result, count(*) from decisions where session_id = $LAST and reason in ('hedge','close_hedge') group by risk_result"` | only `ok` (and `none`); no `delayed`, `no_account`, `pdt` | |
| Ctrl-C mid-run | Ctrl-C after at least one hedge; `q "select status, final_shares from sessions where id = $LAST"`; `q "select id, state from orders where session_id = $LAST and state in ('PENDING','SUBMITTED','PARTIAL','UNKNOWN')"` | exit 0; status `stopped`; no open order rows; TWS Portfolio unchanged (call, put, `final_shares` shares); TWS Orders tab shows no `qmpc:*` order | final_shares = |
| Restart recovers, no second straddle | `python scripts/run_paper.py --execute --contracts 1 --force-open` again; `q "select count(*) from orders where session_id = $LAST and reason = 'entry'"` | log `ledger: M stored fills replayed, realized …` (no `do not rebuild the broker position` WARNING) and `started paper: 1 straddles, N shares, vol sessions K` with N = `final_shares` above; entry count 0; hedging resumes on the first tick with a band crossing | |
| Forced disconnect | TWS → Global Configuration → API → Settings → untick "Enable ActiveX and Socket Clients", OK; wait 60–90 s; re-tick | log `RUNNING --degrade--> DEGRADED`, rows `reason = disconnected`; then `reconnected after N attempt(s); reconcile pending` and `DEGRADED --recover--> RUNNING`; no order placed in between (`q "select count(*) from orders where session_id = $LAST and created_at > '<disconnect ts>'"` → 0 until recovery); a zero-drift reconcile leaves no row marker (`reconcile_adopt` appears only if TWS and the book disagreed) | attempts = |
| Recovery within the halt budget | `q "select ts, reason from decisions where session_id = $LAST and reason like 'halt:%'"` | no rows (the outage was < 600 s) | |
| Flatten | Ctrl-C, then `python scripts/run_paper.py flatten --execute`; `echo $?` | log `flatten attempt 0: 1/1 @ … FILLED`, then the share order; console `flat`; exit 0; TWS Portfolio shows no SPY position; `q "select mode, status from sessions where id = $LAST"` → `flatten flatten` | exit = |

## 4. Sign-off

| Item | Value |
|---|---|
| Date | |
| TWS build (Help → About) | |
| ib_async version (`python -c "import ib_async; print(ib_async.__version__)"`) | 2.1.0 expected |
| `ratio_vega` / `ratio_theta` recorded | |
| `greeks.ibkr_vega_per_volpt` / `ibkr_theta_per_day` set to | |
| Paper NLV used | |
| `max contracts by risk fractions` at that NLV | |
| Straddle entry avg price / commissions seen | |
| Hedges sent in the hour / band_h range | |
| Disconnect recovered in (attempts, seconds) | |
| Signed | |

Only after every row above passes: `python scripts/run_paper.py --execute --live --port 7496`,
without `--force-open` (refused), and type `CONFIRM` when the limits line matches what you expect.
