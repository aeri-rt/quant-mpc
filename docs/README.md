# Documentation

```
docs/
├── decisions/   # Architecture Decision Records (ADRs)
├── design/      # the one design document that remains (owner's intent, with a correction header)
├── guides/      # setup
├── runbooks/    # operating the paper/live loop, paper checklist
├── research/    # paper notes and PDFs
└── templates/   # ADR, design doc, runbook templates
```

## Quick links

### Start here
- [ADR-004: V2 thesis, hedger, time base, safety](decisions/004-v2-thesis-and-hedger.md) — what the system is and why; includes the canonical units table
- [Setup guide](guides/setup.md) — venv, dependencies, TWS settings, `check_ibkr.py`
- [Runbook](runbooks/runbook.md) and [paper checklist](runbooks/paper-checklist.md) — start/stop, reading the system, HALT handling, cost calibration, sign-off before `--live`

### Decisions
- [ADR-001: Project inception](decisions/001-project-inception.md) — historical; superseded by ADR-004 (do-mpc/IPOPT, MHE and `ib_insync` are gone)
- [ADR-003: Data source selection](decisions/003-data-source-selection.md) — IBKR as data + execution; superseded in part by ADR-004 (the client library is now `ib_async`)
- [ADR-004: V2 thesis and hedger](decisions/004-v2-thesis-and-hedger.md) — supersedes ADR-002 and the deleted MPC controller design docs

### Design
- [Gamma scalping V2 alpha spec](design/gamma-scalping-v2-alpha-spec.md) — the owner's intent document, kept for reference; its header lists what ADR-004 corrected (inverted deadband formula, CampaignManager/OptionSelector/MPCScheduler replaced by `gate.py` + `runner.py`, MPC = closed-form QP)

### Research
- [Clinet 2021 paper note](research/papers/clinet-2021-mpc.md) — what the execution paper is, what was taken (the QP-per-step pattern) and what was not, plus the references the hedger actually rests on

## Document types
- **ADRs** in `decisions/`, named `NNN-short-title.md`, from [DECISION_RECORD.md](templates/DECISION_RECORD.md). Any substantial architectural change gets one.
- **Design documents** in `design/`, from [DESIGN_DOC.md](templates/DESIGN_DOC.md).
- **Runbooks** in `runbooks/`, from [RUNBOOK.md](templates/RUNBOOK.md).

Keep links relative and keep the README status table in sync when a row's evidence changes.
