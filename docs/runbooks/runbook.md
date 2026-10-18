# Runbook: quant-mpc paper / live operation

> **Dedicated account.** `Book.from_positions` adopts **every SPY share in the account** as the
> hedge, and an `--execute` start that finds shares but no straddle **flattens them on the first
> RUNNING tick**. Run the bot only in an account (or sub-account) that holds nothing else in SPY.
> A manual trade in SPY is seen only by the next reconcile (≤ `loop.reconcile_s`): SPY option legs
> → `halt:leg mismatch` (execute; dry-run adopts), SPY shares → `reconcile_adopt` (≤ 50) or
> `halt:share drift`; other symbols are ignored.

## Overview

Operating `scripts/run_paper.py` (dry-run by default, `--execute` on the paper port, `--live` on
7496), reading the log and `data/trades.db` while it runs, what a HALT means and how to recover,
liquidation with `flatten`, and the maintenance that keeps the cost model honest. The backtest is
in the README; the design is ADR 004. Version facts: ib_async 2.1.0, Python 3.13, the offline
suite (`pytest -q`, about 20 s).

Runner states: `STARTING → RUNNING ⇄ DEGRADED → HALTED → STOPPING → STOPPED`.

- **DEGRADED** = the SPY quote is older than `loop.stale_quote_s` (15 s, measured as now − the
  ticker's arrival stamp, no clock offset) or the API socket is down — a lost link anywhere in a
  tick (`BrokerDisconnected`: reconcile, account, chain, whatIf, order placement, cancel) is
  DEGRADED, never a halt. No orders are sent; fills are still drained; the reconcile runs while
  the socket is up (skipped while it is down; the reconnect schedules one). A fresh quote moves it
  back to RUNNING, and the recovery first cancels any order the outage left working (`recover: N
  orders left working through the outage cancelled`), then runs the reconcile a reconnect
  scheduled before any decision (fills the outage hid are in `positions()`, not in the book).
- **HALTED** after: stale for > `loop.stale_halt_s` (600 s, a disconnect counts) **while the market
  is open** (after hours or on a holiday a missing quote is DEGRADED `stale` and never halts; the
  timer restarts at the open); daily loss
  (min of IBKR `dailyPnL` and book P&L since session start) < −2 % NLV for 3 consecutive ticks and
  not before 09:35 ET; option-leg mismatch at reconcile; share drift > `risk.max_share_drift`
  (50); 3 order rejects in 10 min; a partial BAG with unequal leg fills ("unwind needed"); an exit
  that still has units after the touch; unpaired legs at start (execute); any other exception in a
  tick.
  **A halt cancels our open orders and never flattens**: bad data is a halt cause and trading on
  bad data is worse than holding a bounded book (long straddle + at most n·100 ± 150 shares).

## Prerequisites

### TWS settings (once per machine)

| Setting | Where | Why |
|---|---|---|
| Enable ActiveX and Socket Clients, port **7497** (paper) / **7496** (live) | TWS → Global Configuration → API → Settings | the only transport |
| **Master API client ID = `broker.client_id` (10)** | same page | orders left by a crashed session are visible to the restart, which rebinds every order whose `orderRef` starts with `qmpc:`; an `--execute` restart then cancels them, a dry-run only logs them (Recovery expectations) |
| Read-Only API **off** | same page | `--execute` places orders; read-only TWS rejects them (`flatten` too) |
| IBKR Pro **Tiered** pricing | Client Portal → Settings → Pricing | `costs.*` defaults assume Tiered ($0.0035/share, $0.35 min, $0.65/contract) |
| OPRA (US options) + Network B (NYSE Arca/AMEX, where SPY prints) | Client Portal → Market Data Subscriptions | live quotes; on delayed SPY data `--execute` refuses to hedge (`risk_result = delayed`), and on delayed SPY or candidate-leg data it opens no entry (`reason = delayed`) and no roll re-entry (the roll exits and the row reads `reason = exit:roll_refused`; the log says `roll re-entry refused (delayed): exited`); exits still go out |
| "Share real-time market data subscriptions with paper trading account" **on** | Client Portal → Settings → Paper Trading Account | otherwise the paper feed is delayed (md type 3) |
| **Reset paper NLV to the intended live size** | Client Portal → Settings → Paper Trading Account | every risk limit is a fraction of NetLiquidation; on the $1M paper default nothing ever binds. 5 straddles ≈ $10k premium at 15 % IV (≈ $14k at 20 %) ⇒ NLV ≈ $70–95k at `max_premium_frac_nlv` 0.15 |
| Weekly 2FA, daily TWS restart | TWS login | both are manual: the bot guarantees **recovery after a restart, not continuity through one** |

`connect(readonly=True)` in dry-run only skips `reqOpenOrders`; it is not what prevents orders —
the SimBroker order path is. `broker.market_data_type: 3` is a request ("delayed when nothing
better exists"); every quote and decisions row carries the type that actually arrived (`md_type`).

### First-session check: `python scripts/check_ibkr.py`

Read-only, never places an order (a whatIf is not an order). Every step prints `[ok]`/`[FAIL]`
with the measured value; exit 1 on any FAIL. Run it with TWS open, during regular hours.

| Line | ok means | FAIL means / fix |
|---|---|---|
| `[FAIL] connect: …` | — | TWS not running, wrong port, API disabled, client id in use |
| `clock_offset_s: +0.12 (market_data_type requested=3)` | \|offset\| ≤ 2 s vs `reqCurrentTime` | sync the host clock (`sudo sntp -sS time.apple.com` on macOS). The runner WARNs above 2 s and **refuses quotes in execute mode above 10 s** |
| `SPY quote: bid=… ask=… age_s=… md=…` | a fresh two-sided quote arrived within 10 s | `none within 10 s`: feed down or no subscription. `md=3` prints `[WARN] delayed data: dry-run works; --execute refuses entries and hedges` |
| `chain: N quotes, M two-sided strikes, expiry YYYY-MM-DD` + `ATM K=… call.mid=… put.mid=…` | one expiry (with `position.prefer_monthly: true` the 3rd-Friday monthly within ±5 days of `entry_dte` 30, else the nearest; never < `roll_dte`+1), ≥ 1 strike quoted on both rights | no OPRA, or every option quote failed validation (one-sided, spread/mid > 10 %, stale) |
| `leg_quote: bid=… ask=… iv=…` | the streamed ATM call has a usable quote | `none after 15 s`: the runner cannot exit a held leg without it |
| `ibkr delta=… vega=… theta=…` / `bs delta=… vega=…/volpt theta=…/day` (`leg_greeks` appears only on FAIL) | tick-106 modelGreeks arrived | `no modelGreeks after 15 s`: OPRA/Greeks not streaming |
| `greeks units: ratio_vega=… ratio_theta=…` | both ratios classified (see below) | one ratio outside both bands: keep the flags `null` (BS vega/theta are used) |
| `whatIf 1 straddle: initMarginChange=…` | a margin number came back | `BrokerError(...)`: BAG not accepted by the paper simulator — see Known limitations |
| `account: NLV=… ExcessLiquidity=… BuyingPower=… DayTradesRemaining=…` + `max contracts by risk fractions: n (why, premium per straddle P)` | account summary readable; `why` is `ok` | `summary unavailable`: account id / permissions |
| `[WARN] NLV … is > 5x the … that max_contracts implies` | — | the $1M paper default: reset the paper NLV (table above) |
| `SUMMARY: all checks passed` / `SUMMARY: FAIL (fix before --execute)` | exit 0 | exit 1 |

**Greek unit flags.** IBKR's vega/theta units are not documented; the script divides them by
Black-Scholes at the same IV. Ratio in [0.7, 1.4] → already $/vol-pt resp. $/day → flag `true`;
ratio in [70, 140] (vega) or [255, 511] (theta) → per unit vol / per year → flag `false`; anything
else → unknown → leave `null` (BS vega/theta). The script prints the two values; set them under
the existing `greeks:` section of `config/live.yaml`:

```yaml
greeks:
  ibkr_vega_per_volpt: true    # or false, from ratio_vega
  ibkr_theta_per_day: true     # or false, from ratio_theta
```

Record both ratios in the paper checklist. The flags only matter with `greeks.source: ibkr`
(IBKR delta/gamma with Γ·dS carry-forward); the shipped default `bs` prices all four Greeks with
Black-Scholes at the tick-106 IV.

**`max contracts by risk fractions: 0`.** The number is `risk.size_contracts` itself, so it
already applies `risk.min_nlv` (25,000; prints `min_nlv`, rows show `size:min_nlv`),
`position.max_contracts` (10) and the premium fraction (`premium_frac`: `floor(0.15 · NLV /
premium)` < 1). Either way the account is too small for one straddle, so raise the (paper) NLV.
Do not raise `risk.max_premium_frac_nlv` to make it fit. The runtime additionally caps it at
`--contracts` / `position.contracts`, and the entry's whatIf margin check can still refuse it
(`straddle:margin`).

## Normal Operations

### Starting the System

```bash
source venv/bin/activate
python scripts/run_paper.py                                   # dry-run: TWS data, SimBroker orders, nothing reaches TWS
python scripts/run_paper.py --execute                         # paper: real orders on broker.port (7497), size from NLV
python scripts/run_paper.py --execute --contracts 1           # first real straddle: fixed 1 (still capped by NLV fraction)
python scripts/run_paper.py --execute --contracts 1 --force-open   # enter with the gate CLOSED (paper checklist only)
python scripts/run_paper.py --execute --live --port 7496      # live: prints the limits, then asks you to type CONFIRM
python scripts/run_paper.py --interval 30                     # override loop.interval_s (also the hedging dt)
python scripts/run_paper.py --client-id 11 --port 4002        # override broker.client_id / broker.port (Gateway paper = 4002)
python scripts/run_paper.py --config config/other.yaml        # another live.yaml (every section required, unknown keys fail)
```

- `run` is the default subcommand; `flatten` is the other one (Emergency Procedures).
- `--force-open` also disables the `exit_confirmed` exit, so a forced campaign holds until roll,
  hard exit (DTE ≤ 1) or max loss (−50 % of premium). It is stamped `force_open = 1` on every
  decisions row and is **refused with `--live`** (`refused: --live with --force-open is refused`).
- `--live` is refused before anything connects unless **both** `--execute` and `--port 7496` are
  on the command line (`refused: --live requires --execute`, `refused: --live requires --port 7496
  on the command line`), then prints `LIVE: real orders, real money. Limits: …` and exits 1 with
  `refused: not confirmed` unless you type `CONFIRM`.
- `mode` on the sessions row is `dry_run`, `paper` (`--execute`) or `live`.

The first console line is the banner, e.g.
`run: mode=paper port=7497 client_id=10 market_data_type=3 interval_s=60 policy=band contracts=1 force_open=True`.
Then, in order: `cancelling stale order <id> BUY 100 STK` for every resting `qmpc:*` order (persisted
to `orders`, then cancelled — `--execute` only, see Recovery expectations); positions → book
(fills that arrived before the snapshot are dropped: `start: N fills before the position snapshot
not booked`; execute mode halts here on `unpaired legs: [conIds]`; dry-run logs it, observes with
an empty book and never reconciles that session); legs and SPY streamed; a held straddle's
ledger replayed from the stored broker fills (`ledger: N stored fills replayed, realized ±X`, or
the WARNING `ledger: N stored fills do not rebuild the broker position; max_loss is measured from
the first tick after this restart`); vol warm-up from 30 sessions of 1-min history (capped at
120 s; a timeout or a link drop logs `vol warm-up failed (TimeoutError()): continuing without it`); `started paper: 1 straddles, 37
shares, vol sessions 29` (today's session is fetched but completes only at the close, so 29
during the session and 30 when started after the close); `STARTING --started--> RUNNING`.
One decisions row per `loop.interval_s` from then on.

A flat book enters only when the gate is OPEN (or `--force-open`), the market is open, more than
`loop.no_entry_last_s` (1800 s) remain before the close, the next session is not in
`instrument.ex_dividend_dates`, no share order is working and, in `--execute`, the SPY quote and
the candidate legs' quotes are live (`md_type` 1). With the gate OPEN (or forced), a blocked entry stamps `reason` `late`,
`ex_div`, `working_order` or `delayed`. A roll's re-entry passes the same guards; a blocked
re-entry leaves the book flat after the exit (`reason = exit:roll_refused`, log `roll re-entry
refused (<why>): exited`).

### Stopping the System

Ctrl-C or `kill -TERM <pid>` = clean stop: `stop requested: signal`, `RUNNING --stop--> STOPPING`,
our open orders cancelled, a final reconcile and a `reason = shutdown` decisions row, session
status `stopped`, `STOPPING --stopped--> STOPPED`. **Positions stay in place** (straddle and
shares); the next start recovers them by conId and never opens a second straddle.
A stop during start-up is honoured between every step and before any order.

Exit codes: **0** STOPPED; **3** state HALTED or `risk.halted` (`HALTED: <reason>` on stderr,
also when the stop arrived after the halt — the session is stored `halted`); **1** anything else
(refused flags, `flatten` without `--execute`, a crash before STOPPED).

### Monitoring

Log: console at `log.level` (INFO) plus `logs/quant-mpc_<date the process started>.jsonl`
(loguru-serialised, DEBUG, 10 MB rotation, 30 days retention; the date is set at start and at each
10 MB rotation, not daily). Components: `runner`, `broker`, `risk`, `store`.

```bash
LOG=$(ls -t logs/quant-mpc_*.jsonl | head -1)
tail -F "$LOG" | jq -c '{t:.record.time.repr, l:.record.level.name, c:.record.extra.component, m:.record.message}'
```

DB: `data/trades.db` (SQLite, WAL), tables `sessions`, `orders`, `fills`, `decisions`. Read it
with the `sqlite3` CLI (or `pd.read_sql_query(sql, store._db)` from Python); **never write to it**.

```bash
DB=data/trades.db
LAST="(select id from sessions order by started_at desc limit 1)"
q() { sqlite3 -header -column "$DB" "$1"; }

# last 20 decisions (one row per tick, newest first)
q "select ts, S, shares, delta_net, band_h, band_u, qp_u, naive_u, u_sent, risk_result, gate_state, round(edge_day,1), reason from decisions order by id desc limit 20"

# fills of the current session
q "select ts, con_id, qty, price, commission, source from fills where session_id = $LAST"

# halts and the sessions they ended
q "select ts, reason from decisions where reason like 'halt:%' order by id desc limit 10"
q "select id, started_at, ended_at, mode, status, final_shares, realized from sessions order by started_at desc limit 5"

# orders of the current session (STK hedges and BAG straddles alike)
q "select id, sec_type, action, qty, limit_price, reason, state, filled, avg_price, error from orders where session_id = $LAST order by created_at"
```

Decisions columns: `ts mode md_type quote_age_s clock_offset_s S bid ask iv_atm sigma_f delta_opt
gamma theta_day vega shares delta_net band_h qp_h band_u qp_u naive_u u_sent policy risk_result
gate_state edge_day leland_a campaign reason nlv daily_pnl book_pnl force_open fill_source`.
`band_u / qp_u / naive_u` are the three hedgers' answers every tick (shadow-logged), `u_sent` is
what went out under `policy` after `risk_result`; all four are NULL on a flat book (the hedge step
only runs with legs or shares). `gate_state` is the only gate column; the gate's own reason
(`gate_open`, `edge_below_min`, `vol_margin`, `cooldown`, `holding`, `exit_confirmed`) is not
stored; a completed exit is the row's `reason = exit:<reason>`, the BAG SELL's `orders.reason`
(`hard_exit`, `roll`, `roll_refused`, `exit_confirmed`, `max_loss`, `flatten`) and the log line
`<reason> attempt N: k/n @ limit STATE`.

Normal ranges: one row per `interval_s`; `quote_age_s` a few seconds on a live feed (> 15 s is
DEGRADED); `clock_offset_s` < 2; `md_type` 1; `gate_state` CLOSED most days with `edge_day`
negative; `band_h` ≈ 30–60 shares at 5 straddles (≈ 15–25 at 1 straddle and 30 DTE, ≈ 20–35 near the roll;
floor `band_min_shares` 5);
`sigma_f` NULL only until 5 completed sessions of realized vol exist.

**`reason` codes** (one per row; a campaign reason set first wins over the hedger's, except
`working_order`, `reconcile_adopt` and `halt:` which overwrite):

| reason | meaning |
|---|---|
| `in_band` | \|Δ\| ≤ band half-width under the active policy; nothing to do |
| `hedge` | the policy wanted a trade; `u_sent` says whether risk let it out |
| `close_hedge` | inside `loop.close_window_s` (300 s) of the close: hedge to flat delta once per session; it counts only once it FILLS — a close hedge cancelled after `fill_timeout_s` or only partly filled is re-sent next tick inside the window |
| `nan` | a non-finite hedger input (Δ, Γ, S, T); zero hedges |
| `no_vol` | flat book: no realized-vol forecast yet (< 5 completed sessions), gate not evaluated; held book under `policy: qp`: no horizon variance, QP output 0 |
| `increases_exposure` | a hedger output would have raised \|Δ\| (bug guard; zeroed) |
| `working_order` | a share order is still open; nothing new sent (no hedge, and no entry or roll re-entry); cancelled once older than `loop.fill_timeout_s` (20 s) |
| `delayed` / `late` / `ex_div` | entry refused (a blocked roll re-entry is `exit:roll_refused`): `--execute` on delayed SPY or candidate-leg data (`md_type` ≠ 1) / ≤ `loop.no_entry_last_s` (1800 s) before the close / the next session is in `instrument.ex_dividend_dates`. Exits are never refused for these |
| `entry` | the entry BAG filled (log `entered k straddles K=… @ …`) |
| `exit:<reason>` | a completed exit: `exit:max_loss`, `exit:hard_exit`, `exit:exit_confirmed`, `exit:roll_refused` (gate CLOSED for the new candidate, or the re-entry blocked by a guard above). A roll's `exit:roll` is overwritten by its re-entry's outcome; `flatten` writes no decisions row except the `halt:exit failed flatten: k units left on (…)` row when units remain after the touch |
| `roll` | a completed roll: exit and re-entry both filled. A re-entry that fails after the exit shows its own reason (`size:…`, `straddle:…`, `whatif_failed`, `entry_unfilled`) on a flat book |
| `stale` / `disconnected` | no fresh quote / API socket down → DEGRADED; no decisions this tick, fills drained, reconcile runs only while connected |
| `greeks_nan` | a held leg has no IV from any source; no decisions this tick |
| `unpaired` | legs that do not form one straddle (dry-run only; execute halts) |
| `no_chain` / `no_iv` | no two-sided ATM pair within 5 % spread on the candidate expiry / no IV for it |
| `spread_wide` | exit/roll/flatten refused: a held leg has no usable quote (one-sided, > 10 % spread, stale); retried next tick |
| `entry_unfilled` | the entry BAG did not fill within `loop.straddle_timeout_s` (30 s) and was cancelled |
| `whatif_failed` | the pre-entry margin whatIf raised; no order |
| `size:<why>` | sizing refused: `no_account`, `min_nlv`, `no_premium`, `premium_frac` |
| `straddle:<why>` | `risk.check_straddle` refused: `halted`, `no_account`, `contracts`, `min_nlv`, `premium_frac`, `margin` |
| `reconcile_adopt` | IBKR shares differed from the book by ≤ 50; adopted (WARNING `reconcile: adopting share drift +N`) |
| `halt:<reason>` | the halt row: `stale quote`, `daily loss L < -M`, `reject storm`, `exception`, `share drift ±N`, `leg mismatch: …`, `unpaired legs: […]`, `exit failed <reason>: k units left on (…)`, `unwind needed: leg fills {…} on order …` |
| `shutdown` | the final row written by a clean stop |

`delayed` appears in both columns: as `risk_result` it is a refused hedge (the row still carries
`S` and the shadow hedges); as `reason` it is a refused entry. **`risk_result`
codes**: `ok`; `none` (u = 0); `halted`; `delayed` (execute mode,
`md_type` ≠ 1); `no_account` (execute mode, account summary failed); `min_nlv`; `pdt`
(`DayTradesRemaining` 0 and NLV < $25k); `increases_exposure`; `book_bound` (\|shares + u\| would
exceed contracts·100 + `risk.max_net_delta_shares` 150 while increasing \|shares\|).

`sessions.status`: `running` (in progress or crashed), `stopped`, `halted`, `flatten` (every
`flatten` run, however it ends).
`fills.source`: `broker` (`--execute`; `commission` NULL until the commissionReport lands, then
patched by execId) or `sim`. `orders.reason`: `hedge`, `close_hedge`, `entry`, the exit reasons
above, `rebound` (adopted from an earlier session, then cancelled).

## Troubleshooting

### Common Issues

#### `risk_result = delayed` on every row, no hedges and no entries (execute mode)
**Symptoms:** `md_type` 3; `[WARN] delayed data` in `check_ibkr.py`.
**Cause:** no Network B subscription, or paper sharing of market data off.
**Resolution:** fix the subscription (Prerequisites); dry-run keeps working meanwhile.

#### `no_vol` rows on a flat book
**Symptoms:** `sigma_f` NULL, gate never evaluated.
**Cause:** fewer than 5 completed sessions of realized vol: the warm-up timed out or the link dropped
(`vol warm-up failed (…)`) or sessions were thin (< half a session of mids are skipped).
**Resolution:** restart during quiet hours so the 30-session history loads; EWMA is used until 60
sessions, log-HAR after.

#### `no_chain` every tick
**Cause:** `chain:` WARNING in the log says which: `no fresh stock quote`, `no SMART option
parameters`, `no expiry with dte >= 8`, or `N contracts, M qualified, none quoted`.
**Resolution:** market closed / OPRA missing / every candidate one-sided. The chain is re-fetched
every 300 s.

#### `working_order` on consecutive rows
**Cause:** the marketable limit (ask + 0.01 / bid − 0.01, DAY) did not fill in 20 s.
**Resolution:** none needed: it is cancelled and re-decided next tick. Persistent: the feed is
behind the market; check `quote_age_s` and TWS's Orders tab.

#### `spread_wide` while trying to exit or flatten
**Cause:** a held leg's quote is one-sided or wider than 10 % of mid (common right at the open
and after 15:50).
**Resolution:** wait; the runner retries every tick. For `flatten`, run it again mid-session.

#### `cancelling stale order …` / `rebound order qmpc:…` at start
**Cause:** the previous session died with a resting order. **Resolution:** with `--execute`, none:
it is persisted and cancelled before anything else happens. **A dry-run does not cancel it**: the
read-only Broker logs `rebound order qmpc:…`, but dry-run orders go to the SimBroker, which never
touches real orders, so the order keeps resting and can fill while you observe. Cancel it with
`python scripts/run_paper.py flatten --execute` (which also liquidates) or by hand in TWS (Orders
tab). Not seeing the line after a crash means the Master Client ID is not 10.

#### `leg <conId>: no tick-106 Greeks, IV frozen at <iv>`
**Cause:** a held leg's modelGreeks (tick 106) are not arriving; its IV stays at the last value
(the restart seed is the leg quote's IV or the implied vol of its mid). Logged at most every 10 min
per leg. **Resolution:** check OPRA / the Greeks stream (`check_ibkr.py` `leg_greeks`); the mid
still comes from the leg quote when it is usable (else Black-Scholes at the frozen IV).

#### `ERROR fill …: unknown conId N, not booked`
**Cause:** a fill of one of the bot's own orders for a conId it never registered (a bug guard;
manual TWS trades never reach this path — the Broker drops executions of orders it did not place
or rebind). **Resolution:** the next reconcile halts on the leg mismatch, see below.

### HALT handling

1. Read the reason: `q "select ts, reason from decisions where reason like 'halt:%' order by id desc limit 1"`
   and the CRITICAL log line `HALT <reason>` (component `risk`). The process has exited with code 3
   and our open orders are cancelled (`halt: N open orders cancelled`). **Positions are untouched.**
2. Decide: hold or liquidate. The book is bounded (long straddle + at most n·100 ± 150 shares)
   and loses theta, nothing more, while you look.
3. Liquidate with `python scripts/run_paper.py flatten --execute` (Emergency Procedures).
4. Deactivation is simply not restarting. A restart is a fresh session that recovers the book by
   conId; the halt flag does not persist, so fix the cause first (feed, clock, account).
5. **Never edit the DB.** The audit trail is the only record of what happened.

Halts that `flatten` cannot handle: `halt:unpaired legs: [conIds]` and `halt:unwind needed: leg
fills {…}` (a lone option leg), and `halt:leg mismatch: …` with a leg count other than one
straddle. `flatten` rebuilds the book with `Book.from_positions`, which refuses a lone leg — it
prints `unpaired legs: [conIds]: unpaired leg, close it by hand in TWS` and exits 1 — and no
single-leg order path exists (the `legged` style is deferred). **Close the odd leg by hand in TWS**
(Portfolio → right-click → Close), then `flatten --execute` handles the remaining straddle and
shares.

### Recovery expectations

**Restart**: with `--execute`, rebound `qmpc:*` orders are cancelled; **a dry-run is read-only and
does not cancel resting real orders a crashed `--execute` session left** — use `flatten --execute`
or TWS (Troubleshooting). IBKR positions → `Book.from_positions` (legs by conId, every SPY share
adopted as the hedge); fills queued before the position snapshot are dropped (the snapshot
already holds them); a held straddle's ledger is rebuilt by replaying the stored broker fills of
its campaign (`Store.campaign_fills`: leg fills since the legs were last flat plus stock fills
since that opening, across sessions; sim fills ignored), so realized P&L, commissions and
`premium_paid` survive. If the replay does not match the broker position: one WARNING `ledger: N
stored fills do not rebuild the broker position…`, `premium_paid` from the broker's average cost
and max_loss measured from the first held tick after the restart. Each leg's IV is seeded from the
first fresh leg quote; `started <mode>: 1 straddles, N shares, vol sessions K`. No entry order is
placed while a straddle is held. In execute mode, shares with no straddle are flattened on the
first RUNNING tick (crash mid-roll recovery).

**Disconnect** (TWS restart, network): `RUNNING --degrade--> DEGRADED` (also when the link drops
inside a broker call: `tick: …: DEGRADED until the link and a fresh quote are back`), rows with
`reason = disconnected` (then `stale` once the socket is back but the ticker has not ticked). No
reconcile while disconnected; an order placed while the socket is down is never sent and is not
counted as a reject. The broker reconnects with backoff 1, 2, 4 … 60 s (`reconnect attempt N
failed: …`, then `reconnected after N attempt(s); reconcile pending`), resubscribes, rebinds
orders, and the runner reconciles on the next tick. A fresh quote gives `DEGRADED --recover--> RUNNING`, which first cancels any order the
outage left working. **Longer than 600 s without a fresh quote while the market is open → `HALT
stale quote`** — a TWS restart during regular hours that takes more than ten minutes (2FA day)
ends the session with exit code 3; restart by hand. Outside regular hours a stale feed never
halts; the 600 s timer starts at the open.

**Reconcile** (every `loop.reconcile_s` 300 s, after a reconnect, at shutdown; deliberately not
after every fill, because IBKR positions can lag the fills, so a mis-booked fill is seen within
300 s; skipped while disconnected and for a dry-run observing an unrecoverable book): share
drift ≤ 50 → adopted with `WARNING reconcile: adopting share drift +N` and `reason = reconcile_adopt`;
larger drift → `halt:share drift +N`; any option-leg difference → `halt:leg mismatch: missing […]
extra […] qty {…}` in execute mode, WARNING and adoption in dry-run. A reconcile with no
difference leaves no marker in the row and no log line. At shutdown a remaining difference is
logged `final reconcile: BookDiff(…)`.

**Idle feed**: 60 s without any inbound message triggers a `reqCurrentTime` probe; only a failed
probe counts as a disconnect (`liveness probe failed: …`).

## Emergency Procedures

### Kill Switch

- From the terminal: **Ctrl-C** (or `kill -TERM <pid>`). Cancels our orders, keeps positions,
  exit 0. This is the fastest way to stop trading.
- Automatic: the halt triggers above, exit 3.
- From TWS: Global Configuration → API → Settings → untick "Enable ActiveX and Socket Clients".
  The bot goes DEGRADED (no orders) at once and halts after 600 s of regular hours. Cancel any `qmpc:*` order you
  see in the Orders tab yourself; the bot would, but it is no longer connected.

### Position Liquidation

```bash
python scripts/run_paper.py flatten --execute        # paper port; add --port 7496 for the live account
```

Connects, recovers the book by conId, cancels our resting orders, refuses outside regular hours
(`market closed: flatten refused`, exit 2), waits up to 10 s for quotes, then **options first**:
BAG SELL at net mid − half-spread, two reprices (−2h, −3h), then the touch (`cq.bid + pq.bid`),
each waiting 30 s (`flatten attempt N: k/n @ limit STATE`); then the shares with a marketable
limit (`orders.reason = flatten`); then re-queries positions and prints `flat` or `residual STK
conId=… qty=…` / `N residual positions`. Exit **0** clean, **1** residual (including `straddle
not closed: hedge shares left in place, retry` when a leg has no usable quote — shares are
deliberately kept while options are open — and the unpaired-leg case above), **2** market closed.
Units still open after the touch halt with `exit failed flatten: k units left on (…)`.
**Ctrl-C / SIGTERM cancel the flatten**: stderr `flatten interrupted: open orders cancelled, check
positions`, exit 1. Every way out (normal return, unpaired leg, market closed, an error, a signal)
cancels our open orders, ends the session row (`mode = flatten`, `status = flatten`) and
disconnects; check TWS Portfolio after an interrupted flatten. There is no auto-flatten anywhere.

### Escalation

There is no operator notification beyond the CRITICAL log line and exit code 3. If the bot runs
unattended, leave this in a second terminal (macOS):

```bash
LOG=$(ls -t logs/quant-mpc_*.jsonl | head -1)
tail -F "$LOG" | grep -m1 CRITICAL && say "halt"
```

(`notify-send` on Linux.) Who to call: the owner; there is no on-call rotation.

## Maintenance

### Regular Tasks

- **Daily:** TWS restarts itself; outside regular hours the bot stays DEGRADED until the feed is
  back and never halts for staleness, during regular hours it either reconnects within 10 min
  (DEGRADED → RUNNING) or halts (exit 3) — check the exit code each morning and restart. Skim
  `q "select id, started_at, ended_at, mode, status from sessions order by started_at desc limit 5"`.
- **Weekly:** 2FA login by hand. Confirm `logs/` is rotating (10 MB, 30 days).
- **Quarterly:** `instrument.ex_dividend_dates` in `config/live.yaml` (SPY goes ex around the 3rd
  Friday of Mar/Jun/Sep/Dec; next ≈ 2026-12-18): no entry is attempted the session before a listed
  date. Bump `exchange_calendars` when a new holiday table ships.
- **After the first `--execute` session, and whenever IBKR changes a fee:** cost calibration.

### Cost calibration

```bash
python scripts/calibrate_costs.py --session <session id>     # id from the sessions table
python scripts/calibrate_costs.py --last 5                   # or the N most recent sessions (default: every session)
                                                             # [--db data/trades.db] [--config config/live.yaml]
```

Reads `data/trades.db` only (never TWS), skips `sim` fills (`skipping N sim fills (charged by the
model itself)`), and compares what IBKR actually charged — `fills.commission`, patched from the
commissionReport — with `src.costs` under the `costs.*` of `config/live.yaml`. Exit 2 with
`run an --execute session first` when there is no DB or no broker fill. Output:

```
broker fills: N (M without a commission report, skipped)
STK n orders (m fills): actual 12.40 modeled 11.95 ratio 1.038 mean |actual-modeled|/actual 4.1%
  buys: effective 0.00412 $/sh (config commission_per_share 0.0035), minimum binds on 60% of fills, rate where it does not: 0.00350
  sells: inferred SEC rate 2.06e-05 (config sec_fee_rate 2.06e-05, finra_taf_per_share 0.0)
OPT k BAG orders: actual 2.70 modeled 2.70 ratio 1.000
config/live.yaml costs: change            # followed by the YAML lines, or
config/live.yaml costs: within 20%, nothing to change
```

- **effective $/share** is actual ÷ shares over buys; the **per-share rate** is read only from
  fills the $0.35 minimum does not bind (a fill at the minimum says nothing about the rate), and
  **minimum binds on x %** is how often it did — at 1 straddle most hedges are < 100 shares, so
  expect a high share.
- the **inferred SEC rate** comes from sells net of the commission model and the (capped) TAF,
  compared with `sec_fee_rate` directly.
- paste the printed YAML lines into `config/live.yaml` when a value is off by more than 20 %.

The same check runs once at runtime: the first real share fill whose commissionReport differs
from `commission() + reg_fees()` by more than 20 % logs `WARNING cost model mismatch:
actual=… modeled=…, run scripts/calibrate_costs.py`.

Schedule facts the defaults encode: FINRA TAF is $0 through 2026-12-31 and **$0.000195 per share
sold from 2027-01-01, capped at $9.79 per trade** — set `finra_taf_per_share` on that date. The
SEC Section 31 rate changes by fiscal year and must be read from the commissionReport, not
assumed. The band's cost `k`, the QP's `c_side` and the gate's hedge-cost term all come from these
numbers, so a stale model mis-sizes every band.

### Updates

```bash
git pull
pip install -r requirements.lock            # pinned; pip check must stay clean
ruff check src backtest scripts tests && pytest -q    # offline suite, ~20 s
python scripts/check_ibkr.py                # again if src/broker.py or src/validation.py changed
python scripts/run_paper.py                 # one dry-run session before the next --execute
```

A config change is validated at start (pydantic, `extra = forbid`, every section required); a bad
key fails before anything connects.

## Known limitations

- **Greek units** are unverified until `check_ibkr.py` has run and the two `greeks.ibkr_*` flags are
  set; until then vega/theta are Black-Scholes. They matter only under `greeks.source: ibkr`; the
  default `bs` prices all four Greeks with Black-Scholes at the tick-106 IV.
- **BAG combos on the paper simulator** are documented by IBKR as "limited". The first paper
  straddle is 1 contract; a rejected BAG shows as `orders.state = REJECTED` with the IBKR text in
  `error`, and three rejects in 10 min halt. There is no `legged` fallback (deferred). If paper
  rejects BAGs, open the straddle by hand in TWS and start the bot with `--force-open`: it
  recovers the legs by conId and hedges them.
- **The gate is closed most of the time**, so live hedging samples are rare. The shadow columns
  (`band_u / qp_u / naive_u`) are filled on every tick with a held book, and `--force-open`
  exists for the paper checklist only.
- **Dry-run on delayed data proves plumbing only** (`md_type` 3 on every row, `fill_source = sim`).
  Dry-run hedges and enters on delayed quotes; execute mode refuses hedges, entries and roll
  re-entries on them (exits still go out).
- **Ex-dividend** (≈ 2026-12-18): no entry the session before a listed date, and nothing else — a
  short hedge held over the date is not warned about, and payment-in-lieu is not booked
  (deferred; ≈ $2 per straddle per quarter).
- **No operator notification** beyond the CRITICAL log line (the `say` one-liner above).
- **Vol forecast** is `None` before 5 completed sessions (EWMA until 60, log-HAR after); sessions
  with less than half their minutes quoted are skipped. Live warms `loop.warmup_sessions` (30)
  sessions at every start (RealizedVol is not persisted), so the gate runs EWMA (log-HAR needs
  `vol.har_min_sessions` = 60 continuous sessions); the backtest warms 60 and uses log-HAR, so its
  gate decisions do not test the live forecaster.
- **Daily TWS restart / weekly 2FA** are manual; a restart longer than 10 min during regular hours
  halts the session.
- **Dry-run never cancels real orders**: a resting `qmpc:*` order from a crashed `--execute`
  session survives a dry-run restart; cancel it with `flatten --execute` or in TWS.
- **Entry cut by a disconnect:** the entry BAG is cancelled on recovery, but legs it already
  filled are booked without being streamed, so their exits return `spread_wide` until the next
  restart re-streams them. Fills the outage hid from the Broker (ib_async does not replay
  re-fetched executions) are caught by the recover reconcile: `halt:leg mismatch` in execute.
- The paper account's `dailyPnL` can lag the book by a tick at the open; the kill switch takes the
  worse of the two, so a stale broker number can only make it stricter.
