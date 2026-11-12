# AutoPost

AutoPost is a local-first reference implementation of an auditable post-training control plane for verifiable mathematical rewards.

Revision 3 contracts and split configuration build on the reproducible Sprint 0 development baseline. Small-model work validates engineering behavior only; it is not evidence of target-model scientific improvement.

## CPU development

Use CPython 3.13.7 in a fresh virtual environment:

```text
make install-cpu
make test
make selfcheck
```

The CPU environment does not download models or install the GPU training stack.

## Toolchain API probe

Create and activate a separate virtual environment, then run:

```text
make install-toolchain-probe
make toolchain-probe
```

The checked-in probe lock reproduces the validated CPython 3.13.7 Windows CPU inspection environment. It validates the pinned TRL/Transformers/PEFT API surface but does not claim that vLLM or a GPU training path has run; those runtime checks remain hardware-gated.

## Design documents

- [Architecture and scientific protocol](post_training_documentation.md)
- [Implementation sprint roadmap](implementation_sprint_roadmap.md)
- [Contracts, configuration, and compatibility](docs/CONTRACTS.md)
- [Sprint 1 acceptance evidence](docs/SPRINT_1_AUDIT.md)
- [Durable stores and fixture execution](docs/STORES.md)
- [Sprint 2 acceptance evidence](docs/SPRINT_2_AUDIT.md)
- [Data pipeline and hidden-development workflow](docs/DATA.md)
- [Sprint 3 acceptance evidence](docs/SPRINT_3_AUDIT.md)
- [Training verification, curriculum, and label reviews](docs/VERIFICATION.md)
- [Sprint 4 acceptance evidence](docs/SPRINT_4_AUDIT.md)

Inspect the composed development configuration and identities:

```text
python -m autopost.cli config show
python -m autopost.cli config hashes
python -m autopost.cli config placeholders
```

`configs/base.yaml` contains experiment semantics and selects separate model, execution, and data files. `--model`, `--execution`, and `--data` override these selections; relative paths resolve beside the base file. For example, `config hashes --execution execution/smoke.yaml` selects the fixture profile. This selects configuration only; backend execution arrives in later sprints.

Exercise the durable architecture without a model or GPU:

```text
python -m autopost.cli fixture run
python -m autopost.cli fixture run --scenario duplicate_success
python -m autopost.cli fixture run --scenario conflict
python -m autopost.cli db status --path runs/sprint2-fixture/success/metadata.sqlite
```

Fixture reports and artifacts live under `runs/sprint2-fixture/<scenario>/`. Repeating a successful fixture returns a duplicate and preserves its original state and budget charge. Use `python -m autopost.cli db migrate --path <database>` to apply checksum-verified additive metadata migrations; existing scientific records remain unchanged.

Build the disposable CPU data fixtures and inspect their contamination and review reports:

```text
python -m autopost.cli data build-dev
python scripts/build_hidden_perturbations.py
```

The first command publishes separate immutable training and measurement datasets under `runs/development-data/`. Hidden-program samples remain quarantined until explicit human review. No research `hidden_audit`, `hidden_sealed` or sealed transfer dataset is created. The [data guide](docs/DATA.md) covers pinned local source snapshots, cached embeddings and review commands.

Sprint 4 adds a pinned CPU verifier dependency to `make install-cpu`.
`python -m autopost.cli training --help` exposes rollout verification, curriculum
reservations, promotion refresh and independent label review. The
[training-signal guide](docs/VERIFICATION.md) covers inputs and evidence.
