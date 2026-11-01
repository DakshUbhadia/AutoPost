# Dependency and model pins

## Status (2026-10-08)

Pins are intentionally unresolved. The implementation brief requires the full environment lock to be generated on the Phase I GPU machine because a CPU-only environment can select incompatible `+cpu` Torch wheels. The target GPU machine and OS have not yet been supplied. No package versions or Hugging Face commit IDs have been invented.

| Item | Pin | Verification | Status |
|---|---|---|---|
| Direct Python dependencies | `pyproject.toml` | PyPI metadata and compatibility checks required by T4 | Pending target machine |
| Transitive dependencies | `requirements.lock` | Resolve with uv/pip on target GPU machine | Pending target machine |
| TRL `GRPOConfig` spellings | — | Required field and constructor check from T4 | Pending resolved TRL |
| `Qwen/Qwen3-0.6B` | — | Hugging Face model commit lookup | Pending T5 |
| `Qwen/Qwen3-4B-Instruct-2507` | — | Hugging Face model commit lookup | Pending T5 |
| `BAAI/bge-small-en-v1.5` | — | Hugging Face model commit lookup | Pending T5 |

Do not run `make install` until the placeholders in `pyproject.toml` have been replaced and `requirements.lock` has been generated.
