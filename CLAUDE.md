# Agent guide

- P1: Separate training and measurement channels with independent scoring, templates, splits, transfer data, and hidden labels.
- P2: Keep the champion immutable; save full resumable checkpoints.
- P3: Use deterministic checking; no learned judge decides outcomes.
- P4: Nothing uncalibrated decides; thresholds are pre-registered or derived from Phase I measurements.
- P5: The controller halts rather than self-repairing; human repairs consume a budget and refresh Gate-A.
- P6: Count and budget every measurement look; read the claim set once per run.
- P7: Keep data assets durable; checkpoints may be pruned.
- P8: Add extensions through existing interfaces only when a measurable trigger fires.
- P9: Validate power, false promotion, bootstrap coverage, and false halts before trusting verdicts.
- P10: Set measurement cadence and training budgets from the observed learning curve.

Run `make selfcheck` for synthetic checks and `make test` for the fast test suite. See `docs/architecture.md`, `docs/PINS.md`, `docs/PLACEHOLDERS.md`, `docs/PREREG_DECISIONS.md`, and `docs/OPEN_ISSUES.md`.

`exact_scorer.py` is standard-library only. Measurement items never reach a trainer. Thresholds are pre-registered, never tuned after seeing results.

Do not install until dependency pins and `requirements.lock` are resolved for the Phase I GPU machine.
