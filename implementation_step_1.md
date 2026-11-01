# Implementation Step 1 — Repository, dependency pins and pre-registration

> **Naming note.** This is the *first* build step. In the roadmap it is labelled "Step 0: Setup and pre-registration". The next step in the roadmap ("Step 1: compatibility spike", `scripts/spike_model.py`) is **not** part of this document.

**Audience:** an AI coding agent working in a fresh git repository, with a shell and (ideally) network access to PyPI and Hugging Face.
**Estimated effort:** 2–3 working days, plus the time a human needs to make the pre-registration decisions in T11.
**Source of truth:** `post_training_documentation.md` (Revision 2, "the architecture doc"). Where this plan and the architecture doc disagree, stop and ask the human.

---

## 0. Rules for the implementing agent

1. **Never invent a human judgement.** The floor, guardrail margins, compute budget and second learning rate (Section 6) are decisions for the human. Build everything around them, test with fixtures, then stop and ask (T11).
2. **Never invent a version or a commit sha.** Every pin must come from a real lookup (PyPI, `pip index versions`, `huggingface_hub`, or the vendor repo page). If you cannot reach the network, stop and ask the human to supply the values.
3. **Do not change the semantics of the companion files.** The only permitted edits are listed in T2, T3, T5 and T11. Anything else you find goes into `docs/OPEN_ISSUES.md` (T10), not into the code.
4. **Never weaken a test or an assertion to make it pass.** If `selfcheck.py` fails for a reason other than import paths, report it to the human with the failing output. The architecture doc states these checks pass, so a real failure is important news.
5. **`exact_scorer.py` must stay standard-library only.** It must never import `math_verify`, `sympy`, `latex2sympy2*`, or any module with "train" in its name (architecture doc §7 P1, §16.2).
6. **Commit after each task** with a message `step1/Tn: <summary>`. Do not commit `runs/`, `data/`, or any secret.
7. **Do not start work that belongs to later steps** (spike, ingest, verifiers, trainer, eval runner, controller). Creating empty placeholder packages is fine; implementing them is not.
8. **Stop and ask the human** (use the interactive question tool if you have one) when: network access to Hugging Face is missing; the pinned TRL release lacks a required parameter (Section T4); the GPU hardware is unknown; any Section 6 value is missing; a companion file looks wrong.

---

## 1. Goal and definition of done

**Goal:** turn the six loose design files into a working, pinned, tested Python repository, and freeze the Phase I pre-registration so that every later experiment is judged against thresholds fixed in advance (architecture doc P4, D25).

The step has two parts.

**Part A — agent-completable (T1–T10).** Done when all of these hold:

- [ ] `make install` creates a working environment from the pinned dependencies.
- [ ] `make selfcheck` (runs `scripts/selfcheck.py`) exits 0 in the real repo layout.
- [ ] `make test` passes (fast tests only).
- [ ] `configs/base.yaml` has no `<pin ...>` placeholders left, and every remaining `null` is inventoried in `docs/PLACEHOLDERS.md` with its owner (an experiment, or the pre-registration).
- [ ] `docs/PINS.md` records every pinned version/sha, how it was verified, and the TRL parameter spellings valid for the pinned release.
- [ ] `python -m autopost.cli prereg validate/freeze/verify` work and are covered by tests that use **fixture** values (never real ones).
- [ ] `docs/PREREG_DECISIONS.md` lists every human decision with guidance, ready to put to the human.

**Part B — human-gated (T11).** Done when:

- [ ] The human has supplied the Section 6 values.
- [ ] `configs/prereg/stageB-p1-v1.json` is frozen (read-only), its sha256 is in the `prereg` table of `runs/autopost.sqlite`, `prereg verify` passes, and the git tag `prereg-stageB-p1-v1` exists.

---

## 2. Facts the agent needs

### 2.1 What exists

Six files, all authoritative for their purpose: `base.yaml`, `schema.sql`, `interfaces.py`, `stats.py`, `exact_scorer.py`, `selfcheck.py`, plus the architecture doc. They were validated only on **synthetic** data. No other code exists.

### 2.2 Where each file goes

| Companion file | Destination | Notes |
|---|---|---|
| `post_training_documentation.md` | `docs/architecture.md` | Copy unchanged |
| `base.yaml` | `configs/base.yaml` | Edited in T5 and T11 only |
| `schema.sql` | `src/autopost/store/schema.sql` | **Never edit.** Shipped as package data |
| `interfaces.py` | `src/autopost/interfaces.py` | Already has a `try: from autopost.eval.stats ... except ImportError: from stats ...` fallback; keep it |
| `stats.py` | `src/autopost/eval/stats.py` | No intra-package imports; needs numpy |
| `exact_scorer.py` | `src/autopost/verify/exact_scorer.py` | stdlib only |
| `selfcheck.py` | `scripts/selfcheck.py` | Imports edited in T3 |

### 2.3 Target tree for this step

```
autopost/
├── pyproject.toml            # exact == pins for direct deps
├── requirements.lock         # full transitive pins (T4)
├── Makefile  .gitignore  .python-version  README.md  CLAUDE.md
├── configs/
│   ├── base.yaml
│   └── prereg/               # stageB-p1-v1.draft.json, then stageB-p1-v1.json (frozen)
├── docs/
│   ├── architecture.md  PINS.md  PLACEHOLDERS.md  PREREG_DECISIONS.md
│   ├── OPEN_ISSUES.md  selfcheck_baseline.txt
├── scripts/
│   ├── selfcheck.py
│   └── check_pins.py         # T4: reusable pin/compat report
├── src/autopost/
│   ├── __init__.py  cli.py  config.py  prereg.py  interfaces.py
│   ├── eval/{__init__.py, stats.py}
│   ├── verify/{__init__.py, exact_scorer.py}
│   └── store/{__init__.py, schema.sql, db.py}
├── tests/
│   ├── conftest.py  test_imports.py  test_selfcheck.py
│   ├── test_config.py  test_db_init.py  test_prereg.py
├── data/   (gitignored)
└── runs/   (gitignored)
```

This follows architecture doc §29.1. It adds `config.py`, `prereg.py`, `scripts/check_pins.py` and `docs/`. The other directories from §29.1 (`models/`, `tasks/`, `curriculum/`, `train/`, `gate/`, `controller/`, `obs/`) are created in later steps. Do not create them now.

---

## 3. Tasks

Dependency order: T1 → T2 → T3 first. Then T4+T5 and T6+T7+T8 can proceed in parallel. T9 needs T6–T8. T10 follows. T11 is last.

### T1. Repository skeleton and tooling

1. `git init`. Add `.gitignore` covering `runs/`, `data/`, `.venv/`, `__pycache__/`, `*.sqlite*`, `.pytest_cache/`, and a comment that `data/hidden/` must have restricted permissions. Do **not** ignore `configs/prereg/`.
2. Choose the Python version **after** T4 establishes what the pinned vLLM and TRL support. Start with 3.11 and write it to `.python-version`. Revise if T4 forces a change.
3. `pyproject.toml`:
   - `src/` layout, package `autopost`, setuptools or hatchling.
   - `[project.scripts] autopost = "autopost.cli:main"`.
   - `schema.sql` included as package data.
   - Direct dependencies (exact `==` pins are filled in by T4): `numpy`, `torch`, `transformers`, `peft`, `trl`, `vllm`, `math-verify`, `sympy`, `datasketch`, `sentence-transformers`, `pydantic`, `pyyaml`, `pyarrow`, `huggingface_hub`.
   - Optional group `dev`: `pytest`, `ruff`.
   - pytest config: `testpaths = ["tests"]`, markers `slow` and `gpu`.
4. `Makefile` with working targets `install`, `selfcheck`, `test`, `test-all`, `prereg-verify`. The remaining targets from the architecture doc (`spike data hidden phase1 gonogo run report`) must exist and print `not implemented yet (see roadmap)` and exit 1.
5. Create empty `__init__.py` files for the packages in 2.3.

**Accept:** `git status` clean after first commit; `make` with no target lists the targets.

### T2. Place the companion files and fix imports

1. Copy files per table 2.2. Do not edit `schema.sql`, `base.yaml` (yet), or any logic.
2. Confirm that `from autopost.interfaces import *` works. `interfaces.py` imports `autopost.eval.stats` first, which must exist, so create the package directories before testing.
3. `tests/test_imports.py`:
   - Package form: import `autopost.interfaces`, `autopost.eval.stats`, `autopost.verify.exact_scorer`.
   - Flat form: copy `interfaces.py` and `stats.py` into `tmp_path`, then run a subprocess that does `sys.modules["autopost"] = None; sys.path.insert(0, tmp_path); import interfaces, stats`. Setting the module to `None` forces the `except ImportError` fallback, so this proves the flat layout still works.
   - AST check: `exact_scorer.py` imports only stdlib modules.

**Accept:** `pytest tests/test_imports.py` passes.

### T3. Port and run `selfcheck.py`

Permitted edits to `scripts/selfcheck.py`, and only these:

1. Replace the three flat imports with package imports:
   ```python
   from autopost.verify import exact_scorer as xs
   from autopost.interfaces import (BinaryReward, Completion, DefaultGate, GateSpec,
                                    GuardrailSpec, Outcome, Task, Verdict, band_for)
   from autopost.eval.stats import (BootResult, cluster_se, clusters_needed,
                                    paired_cluster_bootstrap, scale_lora_state_dict,
                                    simulate_rate, wilson_interval)
   ```
2. In `independence_test`, replace `open("exact_scorer.py")` with `Path(xs.__file__).read_text()` (the file is no longer in the working directory).
3. Optional: add `--quick` (argparse) that lowers `n_sim` in `c4_demo` (1500 → 200) and `joint_power_demo` (400 → 50). Default behaviour must stay identical. The full run is the acceptance run.

Then:
- Run `python scripts/selfcheck.py` in full once; save its stdout to `docs/selfcheck_baseline.txt`.
- Compare with the synthetic numbers in architecture doc §21.4 and §21.7. Small differences from numpy/version/seed effects are expected and fine. A large discrepancy (for example the cluster-bootstrap A/A false-positive rate far from ~0.06, or the 22/22-style pass counts changing) goes into `docs/OPEN_ISSUES.md` and to the human.
- `tests/test_selfcheck.py`: load `scripts/selfcheck.py` with `importlib.util.spec_from_file_location`.
  - Fast test (default): `scorer_tests(n_random=500)`, `independence_test()`, `reward_and_band()`, `gate_tests()`, `lora_scale_test()`, `c1_demo()`.
  - `@pytest.mark.slow`: the full `main` path.

**Accept:** `make selfcheck` exits 0; `pytest tests/test_selfcheck.py` (fast) passes.

### T4. Resolve and pin the dependency stack

Goal: one internally consistent, exactly pinned set. **Do not guess versions.**

1. Create a fresh virtualenv (`uv venv` or `python -m venv`). Find candidate versions with `pip index versions <pkg>` or the PyPI JSON API.
2. Resolve in this order, because each constrains the next:
   1. **TRL release.** Must satisfy architecture doc assumption **A2**: GRPO with LoRA, vLLM colocate mode, `loss_type="dapo"`, reward scaling switchable off, truncated importance sampling (TIS) available.
   2. **vLLM** within that TRL release's supported range (read the TRL release's `pyproject.toml`/extras or release notes).
   3. **torch** that the chosen vLLM requires.
   4. **transformers, peft** compatible with the chosen TRL.
   5. **math-verify** (it pulls in `latex2sympy2_extended` and a specific `antlr4` version; let pip resolve and record the result), `sympy`, `datasketch`, `sentence-transformers`, `pydantic`, `pyyaml`, `pyarrow`, `numpy`, `huggingface_hub`.
3. Write `==` pins for the direct dependencies into `pyproject.toml`. Generate `requirements.lock` for the full transitive set (`uv pip compile` or `pip freeze`). Prefer generating the lock on the **GPU machine** that will run Phase I. A CPU-only machine yields `+cpu` torch wheels that are wrong for the target. If the GPU machine is not available, say so in `docs/PINS.md` and ask the human where the lock should be produced.
4. Write `scripts/check_pins.py`. It prints, as Markdown: every direct dependency's installed version; `pip check` result; and the TRL parameter check below. Paste its output into `docs/PINS.md` with the date.
5. **TRL parameter check** (this validates `base.yaml` against the real release; the YAML itself says "check spelling for pinned version"):
   ```python
   import dataclasses, trl
   names = {f.name for f in dataclasses.fields(trl.GRPOConfig)}
   required = ["loss_type", "scale_rewards", "epsilon", "epsilon_high", "beta",
               "num_generations", "num_iterations", "mask_truncated_completions",
               "use_vllm", "vllm_mode", "vllm_gpu_memory_utilization",
               "vllm_importance_sampling_correction", "vllm_importance_sampling_cap",
               "temperature", "top_p"]
   ```
   - Report any missing name. Also construct `GRPOConfig(output_dir=tmp, bf16=False, use_cpu=True, loss_type="dapo", scale_rewards=<value>, ...)` **without** `use_vllm` to confirm `dapo` is an accepted loss type and to find which value turns reward scaling off (`"none"`, `False`, …). Record the working spelling in `PINS.md`.
   - Do not try to start vLLM or load models. That is the next roadmap step.
   - If a parameter is only *renamed*, update the key in `base.yaml`'s `train.trl` block and note it in `PINS.md` (this is the permitted edit). If a feature is **missing** (for example no TIS, no `dapo`), try another release; if none works, **stop and tell the human** (assumption A2 is false; architecture doc §18 describes the fallback).
6. Smoke tests that need no GPU: `import trl, peft, transformers`; `from math_verify import parse, verify` and `verify(parse("$\\frac{1}{2}$"), parse("$0.5$"))` is truthy; `import datasketch, sentence_transformers, pyarrow, pydantic, yaml`. If a GPU is present also run `python -c "import vllm"`; if not, record "vLLM import not verified (no GPU); verified in the spike".

**Accept:** `pip check` is clean; `docs/PINS.md` is complete; fresh `make install` from `requirements.lock` reproduces the environment; the Python version in `.python-version` is supported by every pin.

### T5. Pin model, tokenizer and embedding-model revisions

1. Look up the current commit sha (40 hex chars) of each repo's default branch with `huggingface_hub.HfApi().model_info(repo).sha` (or the repo page). Record the date:
   - `Qwen/Qwen3-0.6B`
   - `Qwen/Qwen3-4B-Instruct-2507`
   - `BAAI/bge-small-en-v1.5` (used by `data.dedup.embed_model`)
2. Edit `configs/base.yaml` (permitted edits only):
   - Replace both `revision: <pin commit sha>` values.
   - Replace `trl_version: <pin exact TRL release>` with the T4 version.
   - Add `tokenizer_revision: <same sha as revision>` to both model profiles, and `max_prompt_tokens: null  # set in spike/E1`. `interfaces.ModelProfile` requires both fields and the YAML currently lacks them.
   - Add the embedding model's revision next to `embed_model` (key `embed_model_revision`).
   - Apply any TRL key renames from T4.
3. Do **not** set `max_new_tokens`. It is intentionally `null` until E1; the spike will use a provisional value.
4. Add the shas to `docs/PINS.md`.

**Accept:** `grep -n "<pin" configs/base.yaml` returns nothing; every revision matches `^[0-9a-f]{40}$` (assert this in `tests/test_config.py`).

### T6. Config loader and placeholder inventory

`src/autopost/config.py` (small; no pydantic model of the whole YAML yet):

```python
def load_config(path: str | Path = "configs/base.yaml") -> dict: ...
def config_hash(cfg: dict) -> str:            # use interfaces.stable_hash
def unresolved_fields(cfg: dict) -> list[tuple[str, str]]:
    """Return (dotted_path, kind) where kind is 'null' or 'pin' ('<pin' substring).
    Lists inside the config are indexed like gate.guardrails[0].margin."""
```

Add CLI `autopost config placeholders` that prints this list.

Then write `docs/PLACEHOLDERS.md` by hand from the `base.yaml` comments, giving each remaining `null` an owner. Use this mapping (verify against the file; fix if the YAML differs):

| Placeholder(s) | Owner |
|---|---|
| `*.rollout_sampling.max_new_tokens`, `*.eval_sampling.max_new_tokens` | E1 (provisional 16,384 for the spike) |
| `max_prompt_tokens` | spike / E1 |
| `data.stage_*.sizes_in_clusters.*` | E1 + E3 (hidden sets: program-level) |
| `verifier_pool.workers` | E1 (GPU idle time waiting on verification) |
| `train.trl.vllm_gpu_memory_utilization` | E1 |
| `eval.k.*` | E3 / E4 |
| `gate.floor`, `gate.guardrails[*].margin` | **Phase I pre-registration (this step, T11)** |
| `controller.gate_every_steps` (G), `controller.step_budget` (S) | E1 + E3, then Phase II pre-registration |
| `controller.read_budgets.{gate_a,gate_b,hidden_audit}`, `controller.human_change_budget` | Phase II pre-registration |
| `tripwires.*.alpha`, `tripwires.pool_inband_min_frac` | Phase II (allocated from the false-halt budget; E5 checks) |
| `tripwires.drift_limits.*` | E1 (absolute limits) |

`tests/test_config.py`: `load_config` works; `config_hash` is stable and key-order independent; no `pin` kinds remain; the set of `null` paths equals the inventory in `docs/PLACEHOLDERS.md` (parse the doc's first column, or keep a Python list in the test and assert the doc mentions each path).

**Accept:** `autopost config placeholders` shows only `null` kinds, each documented.

### T7. SQLite initialisation

`src/autopost/store/db.py`:

```python
def schema_sql() -> str:                  # importlib.resources, package data
def connect(path) -> sqlite3.Connection:  # PRAGMA foreign_keys=ON; journal_mode=WAL; row_factory=Row
def init_db(path) -> None:                # creates runs/ dir; applies schema; refuses if tables already exist
```

Add CLI `autopost db init [--path runs/autopost.sqlite]`.

`tests/test_db_init.py`:
- Every `CREATE TABLE` name found in `schema.sql` exists after `init_db`.
- `journal_mode` is `wal`; `foreign_keys` is on.
- A second `init_db` on the same file raises a clear error (it must not silently wipe data).
- Schema sanity: inserting a `tasks` row with `split='gate_a'` and `measurable=0` violates the CHECK constraint; the same row with `split='train'` succeeds. Supply all NOT NULL columns (`task_id, cluster_id, source_id, source_ver, skill, answer_type, measurable, split, problem, answer`).

**Accept:** `pytest tests/test_db_init.py` passes.

### T8. Pre-registration model, draft generator, validator

`src/autopost/prereg.py`. Use a pydantic model (`Prereg`) with `schema = "autopost.prereg/1"`. Phase I scope only; `--phase phase2` must exit with "Phase II pre-registration is built in a later step".

**Contents of the Phase I pre-registration** (architecture doc D25, §22.2, §29.3, Appendix A). The exact JSON shape is yours to choose, but it must contain:

| Block | Field | Prefill | Source |
|---|---|---|---|
| meta | `version` (`stageB-p1-v1`), `phase`, `stage` (`B`), `created_at`, `code_commit` (git sha if available) | agent | — |
| stack | `base_yaml_sha256`, `lock_sha256`, per-model `hf_repo`/`revision`/`tokenizer_revision`, embedding model + revision | agent computes | R7 traceability; informational, not enforced later |
| judgements | `floor` (fraction, e.g. 0.03 = 3 accuracy points) | **null → human** | §21.2, §22 |
| judgements | `guardrail_margins.{transfer, instruction_following, multilingual_math, format_compliance}` | **null → human** | §21.1, §22.2 |
| judgements | `phase1_compute_budget`: `hardware` (GPU model/count/VRAM), `gpu_hours_total`, `calendar_limit_days` | **null → human** | §4 open questions, A1, §25 |
| judgements | `e1a_learning_rates` | `[2.0e-5, null]` → human picks the second | §18.1 |
| statistics | `alpha_a`, `alpha_b`, `guardrail_alpha` = 0.05 (one-sided); `n_boot` = 4000; `bootstrap_unit` = `cluster`; `k_ref_multiplier` = 2 | agent | §21.2, base.yaml |
| statistics | `per_set_power_target` = 0.90; `guardrail_power_target` = 0.97; `target_joint_power` = 0.80 | agent | §21.7 |
| statistics | derived `nominal_false_promotion` = `alpha_a × alpha_b` (0.0025) | agent computes | §21.2 |
| phase1_plan | E1a: 2 rates × 200 steps on `dev`; E1: 1,000 steps, checkpoint every 50, evaluated on `pilot`; E2, E9 matched to E1's compute | agent | §29.2 |
| go_no_go | (1) signal: GRPO − random-reward gain on pilot, cluster-bootstrap interval excludes 0, point estimate ≥ `floor`, within the E1 step budget; (2) power: P(promote ∣ floor) ≥ 0.80 and P(promote ∣ no effect) ≤ `nominal_false_promotion`; (3) controller: expected false halts ≤ 1.0 and no halt of an improving run; (4) budget: main run + replication + evaluation fits `phase1_compute_budget`'s successor main-run budget | agent (encode as structured fields) | §29.3 |

**Required functions:**

```python
def make_draft(version: str, base_yaml: Path, lock: Path) -> dict        # fills prefills + hashes, leaves human fields None
def validate(prereg: dict, base_yaml: Path) -> list[str]                  # returns problems; [] means valid
def canonical_bytes(prereg: dict) -> bytes                                # json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", UTF-8
```

`validate` must reject:
- any `null` in required fields;
- `floor` or a margin outside (0, 1); α outside (0, 1);
- `base.yaml` still containing `<pin` placeholders, or non-40-hex revisions;
- `base.yaml` disagreeing with the prereg on `gate.floor`, `gate.guardrails[*].margin`, or the α values (T11 copies the human's values into `base.yaml` first, so these must match);
- a `version` that already exists in the database (checked at freeze time).

It should **warn** (not reject) when `floor < 0.01` or a margin `< 0.005`, noting that evaluation cost scales roughly with 1/floor² (§21.7).

**Accept:** unit tests (T9) pass.

### T9. Freeze/verify commands and tests

CLI (stdlib `argparse`; no new dependency):

```
autopost prereg init     --version stageB-p1-v1          # writes configs/prereg/<version>.draft.json
autopost prereg validate configs/prereg/<version>.draft.json
autopost prereg freeze   configs/prereg/<version>.draft.json [--db runs/autopost.sqlite]
autopost prereg verify   [--db runs/autopost.sqlite]
```

**`freeze`** (one transaction):
1. Run `validate`; abort on any problem.
2. Refuse if `version` already exists in `prereg`.
3. Write `configs/prereg/<version>.json` as `canonical_bytes`, then `chmod 0444`.
4. `sha256` of exactly those bytes; insert `(version, sha256)` into `prereg`; insert an `events` row `kind='prereg_frozen'` with `{version, sha256, path}`.
5. Print the sha256 and remind the human to create the git tag.

**`verify`:** for every `prereg` row, re-hash `configs/prereg/<version>.json`; print OK/MISMATCH per version; exit non-zero on any mismatch or missing file.

`tests/test_prereg.py` (use a **complete fixture with obviously fake values**, defined inside the test file; never write fixture values into `configs/`):
- the draft from `make_draft` fails `validate` (human fields are null);
- the complete fixture passes `validate`;
- canonicalisation is independent of key order, so the same content gives the same sha256;
- `freeze` writes a read-only file and a matching DB row, and logs the event;
- a second `freeze` of the same version is refused;
- `verify` passes, then fails after the file is made writable and altered;
- `freeze` fails if `base.yaml` disagrees with the prereg's `floor`.

Use `tmp_path` for the DB and prereg directory (make paths parameters, not hard-coded).

**Accept:** `pytest tests/test_prereg.py` passes.

### T10. Documentation, agent guide and Makefile finish

1. `docs/PREREG_DECISIONS.md`: one entry per human decision in Section 6, each with *what it controls*, *guidance from the architecture doc*, and a concrete question ready to ask. Include the cost trade-off: the architecture doc's synthetic example needs about 711 / 316 / 114 clusters per gate set for a 2 / 3 / 5-point floor (those numbers are illustrations; E1 and E3 produce the real ones).
2. `docs/OPEN_ISSUES.md`: log, without fixing, the issues below (Section 5) plus anything new you find.
3. `CLAUDE.md` (≤ 60 lines) for later agents: the P1–P10 principles in one line each; the rule that `exact_scorer.py` is stdlib-only; "measurement items never reach a trainer"; "thresholds are pre-registered, never tuned after seeing results"; how to run `make selfcheck` and `make test`; where the docs are.
4. `README.md`: what this repo is, how to install, the roadmap steps and which are done.
5. Make sure `make prereg-verify` runs `python -m autopost.cli prereg verify`.

**Accept:** all docs exist and are consistent with what T1–T9 actually produced (file names, commands).

### T11. Human gate — decide and freeze (Part B)

1. Ask the human for the Section 6 values (use an interactive question tool if available, else a plain message). Show the guidance from `PREREG_DECISIONS.md`. Do not proceed on assumptions.
2. Write the floor and the four margins into `configs/base.yaml` (`gate.floor`, `gate.guardrails[*].margin`). These are the only permitted edits at this point. Re-run `make test`.
3. Regenerate the draft (`prereg init`) so the stack hashes include the final `base.yaml` and `requirements.lock`; fill the human values; `prereg validate`.
4. `autopost db init` (if the DB doesn't exist), then `autopost prereg freeze ...`.
5. `autopost prereg verify`; commit `configs/prereg/stageB-p1-v1.json`, `configs/base.yaml`, and `docs/`; `git tag prereg-stageB-p1-v1`.
6. Tell the human: the frozen sha256, the tag, and that any later change to these values requires a new, logged pre-registration version (D25, P4).

---

## 4. Final acceptance checklist (run in a fresh clone)

```
make install
make selfcheck                                  # exit 0
make test                                       # fast tests green
python -m autopost.cli config placeholders      # only documented 'null' entries; no 'pin'
python -m autopost.cli prereg verify            # OK stageB-p1-v1   (after T11)
python - <<'EOF'
import sqlite3; c = sqlite3.connect("runs/autopost.sqlite")
print(c.execute("select version, sha256 from prereg").fetchall())
EOF
git tag --list "prereg-*"
```

Report to the human: the table from `docs/PINS.md`, the selfcheck pass/fail with any deviations from the doc's synthetic numbers, the open issues found, and the frozen pre-registration hash.

---

## 5. Known issues to log (do not fix in this step)

1. **`Task` vs `tasks` table.** The `tasks` table has a `NOT NULL source_ver` column and a stored `measurable` column; the `Task` dataclass has neither `source_ver` nor a stored `measurable` (it is a property). `selfcheck.py` builds `Task` positionally, so adding fields now would break it. To be resolved in the data-pipeline step.
2. **`max_new_tokens: null`.** Left null until E1. The spike needs a provisional value (the vendor's 16,384 for 4B).
3. **`ModelProfile` fields** `tokenizer_revision` and `max_prompt_tokens` were missing from `base.yaml` (added in T5, with `max_prompt_tokens` still `null`).
4. **80 GB GPU assumption (A1)** is unverified until the spike and E1; the hardware answer in T11 should be compared against it.
5. **`selfcheck.py` runtime.** The full run may take several minutes (thousands of simulated bootstrap comparisons); that is why `--quick` exists.
6. **Synthetic numbers** in the architecture doc (gate power, coverage, cluster counts) are illustrations produced by `selfcheck.py`, not evidence about model behaviour.

---

## 6. Human decisions needed (T11)

| # | Decision | Notes |
|---|---|---|
| 1 | **Floor**: smallest accuracy gain worth claiming (fraction) | Drives evaluation cost ~1/floor². Must be large enough that E1 could plausibly show it on a heavily post-trained 4B model |
| 2 | **Guardrail margins** (4): transfer accuracy, instruction following, multilingual math, format compliance | Largest acceptable loss vs the base for each, as accuracy fractions |
| 3 | **Phase I compute budget**: GPU model and count, total GPU-hours, calendar limit | The architecture doc's prior is a few GPU-days for E1+E2 on one 80 GB H100-class card at ~3k-token completions, and GPU-days to a few GPU-weeks for all of Phase I |
| 4 | **Second E1a learning rate** (first is 2e-5) | Two rates × 200 steps on the dev split |
| 5 | **Where the GPU machine is** and whether the lock file can be produced there | Needed for T4 and for A1 |
| 6 | Confirm or change the prefilled values: α = 0.05 one-sided (both gates and guardrails), power targets 0.90 / 0.97 / 0.80, expected false halts ≤ 1 | Defaults come from the architecture doc |

---

## 7. Stop-and-ask triggers (summary)

- No network access to Hugging Face or PyPI.
- TRL release lacks `dapo`, switchable reward scaling, or TIS parameters.
- `selfcheck.py` fails for a reason other than import paths, or deviates strongly from the doc's synthetic results.
- Any Section 6 value missing at T11.
- A companion file appears inconsistent with the architecture doc in a way not listed in Section 5.
