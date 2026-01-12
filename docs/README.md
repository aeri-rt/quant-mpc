# Documentation

This folder contains all documentation for the MPC-based quantitative trading system.

## Structure

```
docs/
├── templates/           # Document templates (ADR, Design Doc, Runbook)
├── decisions/           # Architecture Decision Records (ADRs)
├── design/              # Design documents
├── guides/              # How-to guides and tutorials
└── research/            # Research notes and paper summaries
```

## Quick Links

### Getting Started
- [Setup Guide](guides/setup.md) *(coming soon)*
- [IBKR Configuration](guides/ibkr-configuration.md) *(coming soon)*

### Architecture
- [ADR-001: Project Inception](decisions/001-project-inception.md) - Foundational decisions
- [Architecture Overview](design/architecture-overview.md) - System design

### Research
- [Clinet 2021 Paper Summary](research/papers/clinet-2021-mpc.md) *(coming soon)*

## Document Types

### Architecture Decision Records (ADRs)
Located in `decisions/`. These capture significant architectural decisions with context, rationale, and consequences.

Naming: `NNN-short-title.md` (e.g., `001-project-inception.md`)

Template: [DECISION_RECORD.md](templates/DECISION_RECORD.md)

### Design Documents
Located in `design/`. These describe the design of specific components or features in detail.

Template: [DESIGN_DOC.md](templates/DESIGN_DOC.md)

### Runbooks
Located in `guides/`. These provide operational procedures for running and maintaining the system.

Template: [RUNBOOK.md](templates/RUNBOOK.md)

## Contributing to Docs

1. Use the appropriate template from `templates/`
2. Follow naming conventions
3. Link related documents
4. Keep docs updated as system evolves

