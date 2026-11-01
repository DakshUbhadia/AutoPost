# AutoPost

AutoPost is a research repository for gated continual post-training with verifiable math rewards. The architecture and contracts are documented in `docs/architecture.md`.

## Setup

After the dependency pins and GPU-specific `requirements.lock` are completed, install with `make install`. Run `make selfcheck` and `make test` for local validation.

## Roadmap

| Step | Status |
|---|---|
| Setup, repository skeleton, contracts, config/database/preregistration tooling | In progress; verified pins and human preregistration decisions remain |
| Compatibility spike | Not implemented |
| Data ingest and verifier pipeline | Not implemented |
| Trainer, evaluation, and controller | Not implemented |

The implementation plan is in `implementation_step_1.md`; unresolved choices and pin status are in `docs/`.
