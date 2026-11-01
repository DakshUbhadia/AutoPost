# Open issues (do not fix in setup step)

1. `Task` versus `tasks` table: the table requires `source_ver` and stores `measurable`; the dataclass has neither as fields (`measurable` is a property). Resolve in the data-pipeline step without breaking positional construction in `selfcheck.py`.
2. `max_new_tokens` remains null until E1. The compatibility spike needs a provisional value; vendor guidance for the 4B model is 16,384.
3. `ModelProfile` requires `tokenizer_revision` and `max_prompt_tokens`, currently not present in `base.yaml`. T5 must add these after valid model revisions are looked up; `max_prompt_tokens` stays null until spike/E1.
4. Architecture assumption A1 (80 GB GPU) is unverified until the GPU host is identified and the spike/E1 run.
5. Full `selfcheck.py` may take several minutes. It has a `--quick` mode for development; run the full command for acceptance when authorized.
6. Architecture synthetic power, coverage, and cluster counts illustrate expected behavior and are not evidence about model behavior.
7. T4/T5 pins and `requirements.lock` are pending identification of the Phase I GPU machine. `pyproject.toml` intentionally retains explicit pin placeholders until resolved; installation is not yet reproducible.
8. The requested selfcheck baseline was not generated because this implementation request explicitly prohibits running tests or code checks. Run `python scripts/selfcheck.py` and save stdout to `docs/selfcheck_baseline.txt` when ready.
