# Setup Guide

Development environment plus the TWS settings the paper loop depends on.

## Prerequisites

- **Python 3.12 or 3.13** (`requires-python = ">=3.12,<3.14"`)
- **Git**
- **Interactive Brokers TWS or IB Gateway** with a paper account (port 7497) for anything that touches a broker; the test suite and the synthetic backtest need neither

## Quick setup

Scripts do everything below in one go: `./scripts/setup_macos.sh` (macOS/Linux) or `.\scripts\setup_windows.ps1` (Windows).

### macOS / Linux

```bash
git clone https://github.com/aeri-rt/quant-mpc.git && cd quant-mpc
python3 -m venv venv && source venv/bin/activate
pip install $(python -c "import tomllib;d=tomllib.load(open('pyproject.toml','rb'))['project'];print(' '.join(d['dependencies']+d['optional-dependencies']['dev']))")
pytest -q
```

### Windows (PowerShell)

```powershell
git clone https://github.com/aeri-rt/quant-mpc.git; cd quant-mpc
python -m venv venv; .\venv\Scripts\Activate.ps1
$Deps = & python -c "import tomllib;d=tomllib.load(open('pyproject.toml','rb'))['project'];print('\n'.join(d['dependencies']+d['optional-dependencies']['dev']))"
pip install $Deps
pytest -q
```

The pinned alternative on either platform is `pip install -r requirements.lock`. There is no `pip install -e`: `src/`, `backtest/`, `scripts/` and `tests/` are imported from the repo root (`pytest` sets `pythonpath = ["."]`, scripts insert it themselves).

## Dependencies

Runtime: `ib_async`, `numpy`, `scipy`, `pandas`, `pyarrow`, `pydantic`, `pyyaml`, `loguru`, `exchange_calendars`, `tzdata`. Dev: `pytest`, `pytest-asyncio`, `hypothesis`, `ruff`, `matplotlib`, `py_vollib`. No solver library: the hedger's QP is a closed form (ADR-004).

Check that the pieces import:

```bash
python -c "import ib_async, exchange_calendars, pydantic, scipy; print('ok')"
pip check
```

## Backtest without a broker

```bash
python scripts/run_backtest.py --synthetic --hedger all --force-open   # synthetic bars from backtest.yaml's seed
python scripts/fetch_data.py --cboe                                     # Cboe VIX9D/VIX/VIX3M closes → data/cache
```

Real-bar backtests need `python scripts/fetch_data.py --ibkr --start … --end …` with TWS running (1-min MIDPOINT bars cached per session under `data/cache`), then `python scripts/run_backtest.py --ibkr --start … --end …` (`--ibkr` forces `source: ibkr`; with nothing cached it exits with the `fetch_data.py --ibkr` hint). The first `vol.har_min_sessions` (60) sessions of every window are RealizedVol warm-up only, so fetch ≈3 months of bars before each ≥3-month evaluation window (e.g. `--start 2024-10-07 --end 2025-03-31` trades from 2025-01-02).

## TWS / IB Gateway settings

Set these once in the paper account before the first `run_paper.py`:

1. **API enabled**, socket port **7497** (paper; 7496 is live), trusted IP 127.0.0.1.
2. **Master Client ID = `broker.client_id`** (10 in `config/live.yaml`). Without it orders placed by an earlier session are invisible to a restart, and the start-up cancel of stale `qmpc:*` orders cannot find them.
3. **Share real-time market data with the paper account** (account management → market data subscriptions). Dry-run accepts delayed data (`market_data_type: 3`) and says so in every `decisions` row; unless the SPY ticker reports live data (`marketDataType == 1`), `--execute` refuses to hedge (`risk_result = delayed`), and unless the SPY ticker and the candidate legs do, it opens no entry (`reason = delayed`) and no roll re-entry (the roll exits and the row reads `reason = exit:roll_refused`; the log says `roll re-entry refused (delayed): exited`); exits still go out.
4. **Tiered pricing** (account settings → pricing). `costs.commission_per_share: 0.0035` and `min_commission: 0.35` are the Tiered schedule; Fixed pricing makes every cost number wrong.
5. **Reset the paper NLV to the intended live size.** Every risk limit is a fraction of NetLiquidation; the $1M paper default sizes 10 straddles where a $80k account would size 5. `check_ibkr.py` warns when NLV is more than 5× what `max_contracts` implies.
6. OPRA (options) and Network B (ARCA, for SPY) subscriptions on the account; weekly 2FA is manual.

## First connection

```bash
python scripts/check_ibkr.py            # read-only, never places an order; exit 1 on any [FAIL]
```

It prints `[ok]`/`[FAIL]` for: clock offset (≤ 2 s), a fresh SPY quote and its market data type, a two-sided ATM pair from the one-expiry chain, one streamed leg's tick-106 Greeks against Black-Scholes (the vega/theta ratios decide the `greeks.ibkr_vega_per_volpt` / `ibkr_theta_per_day` flags — set the two printed values under `greeks:` in `config/live.yaml`; ratios outside both [0.7, 1.4] and the ×100/×365 bands stay `null` = Black-Scholes), a BAG `whatIf` margin for one straddle, and the account summary with the contract count `risk.size_contracts` allows (and its reason, e.g. `min_nlv`).

Then, in order: `python scripts/run_paper.py` (dry-run, orders to SimBroker) → `python scripts/run_paper.py --execute --contracts 1 --force-open` → the sign-off list in [docs/runbooks/paper-checklist.md](../runbooks/paper-checklist.md). Operating notes (start/stop, reading the logs and the sqlite tables, HALT handling, `flatten`, cost calibration) are in [docs/runbooks/runbook.md](../runbooks/runbook.md).

## Common issues

- **`pytest` cannot import `src`** — run it from the repo root with the venv active; `pythonpath = ["."]` is in `pyproject.toml`.
- **`exchange_calendars` holiday tables** — the pin is `<5`; bump it when a new year's holidays are missing (XNYS is built once at import, ≈0.3 s).
- **Python 3.14** — not supported yet (`ib_async` open issues); stay on 3.12/3.13.
- **`check_ibkr.py` says `clock_offset_s` FAIL** — fix the host clock (NTP). In execute mode quotes are refused at |offset| > 10 s.
- **`leg_quote: none after 15 s`** — the options market data subscription is missing or the market is closed; the Runner cannot exit a held leg without a two-sided quote.

## Windows-specific issues

- **PowerShell blocks `Activate.ps1`** (`running scripts is disabled`): `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in an elevated PowerShell.
- **Microsoft Visual C++ Build Tools required** during `pip install`: install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) → "Desktop development with C++", then reopen the terminal.
- **Long paths**: set `HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled` to 1 (`New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force` as Administrator) and restart.
- **Python not found**: reinstall from python.org with "Add Python to PATH" checked.
- **Ctrl-C**: `run_paper.py` falls back to `signal.signal` + `call_soon_threadsafe` on the Proactor loop; the stop is still clean (orders cancelled, positions kept).

## IDE

VS Code / Cursor: select the `venv` interpreter (`Cmd/Ctrl+Shift+P` → "Python: Select Interpreter"). PyCharm: Settings → Project → Python Interpreter → Existing environment → `venv/bin/python` (`venv\Scripts\python.exe` on Windows).
