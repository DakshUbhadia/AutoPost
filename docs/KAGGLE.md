# Run Sprint 5 in a Kaggle notebook

Use [the runnable notebook](../notebooks/sprint5_kaggle.ipynb). It contains seven
ordered Python cells and an optional GitHub evidence uploader. No token or
executed notebook output is included in the file.

## 1. Publish the local Sprint 5 source first

The implementation is currently a local working-tree change. The notebook clones
GitHub, so publish that source before running it. In your local PowerShell terminal:

```powershell
Set-Location 'C:\Users\DELL\Desktop\Post-Training'
git status --short
git add Makefile README.md configs docs environments requirements/small-gpu-dev.in scripts/probe_dev_model.py src tests notebooks
git commit -m "feat: implement Sprint 5 execution and Kaggle validation workflow"
git -c credential.username=DakshUbhadia push origin main
```

Review the status before committing. These commands use your existing local Git
credentials. Do not paste your GitHub token into the terminal command or Git URL.
The notebook checks for the Sprint 5 files and stops if the source is missing.

## 2. Configure the Kaggle notebook

1. Keep it private while working. Enable **Internet** and **GPU** in notebook
   settings. Choose T4 if available; the current vLLM backend rejects a P100.
2. Import `notebooks/sprint5_kaggle.ipynb` using Kaggle's notebook import option,
   or copy its seven code cells into your existing notebook in the same order.
3. If you want the optional GitHub upload, use **Add-ons → Secrets** to add a
   secret named exactly `GITHUB_TOKEN`, then enable it for this notebook.
4. Restrict a fine-grained token to `DakshUbhadia/AutoPost` with **Contents:
   read and write**. No administration or workflow permission is needed for the
   results upload. Never put the value into a code cell or printed output.

Kaggle's [secret API](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
retrieves the value only in the final optional upload cell. The uploader uses
GitHub's [Git database API](https://docs.github.com/en/rest/git/trees), which
requires Contents write access for creating the new result tree/commit/branch.

## 3. Run cells 1–5 in order

| Cell | Action |
|---|---|
| 1 | Check NVIDIA GPU/driver, select explicit bf16/fp16 profile, create evidence folder |
| 2 | Clone the published source and record the exact source commit |
| 3 | Create isolated Python 3.13.7, resolve candidate dependencies, install package and check consistency |
| 4 | Run the real subprocess fixture and GPU preflight; require `READY_NOT_RUN` |
| 5 | Run the actual adapter/reload/generation/TRL compatibility campaign |

The cells invoke the virtual environment's Python explicitly, so they do not rely
on activating a shell or replacing Kaggle's notebook kernel. Only GPU 0 is used.
Environment caches and large downloaded packages live in `/tmp/autopost-sprint5`;
evidence lives under `/kaggle/working/sprint5-<run-id>/`.

The dependency input remains an unvalidated candidate. If installation or
preflight fails, stop cells 3–5 and preserve the logs using cells 6–7. Changing
versions or tolerances to force a pass would erase useful compatibility evidence.
For a genuinely new attempt, start again from cell 1 to get a new run ID.

## 4. Save results, including failed runs

Run cell 6 after success **or failure**. It creates
`sprint5-evidence-<run-id>.zip` in `/kaggle/working`, with raw campaign files,
artifacts, logs, package inventories and checksums. Download this ZIP through
Kaggle Outputs and save a notebook version with outputs before ending the session.
Scratch environments and caches are temporary; they are not the evidence bundle.

`PASS` means both GPU phases and `pip check` succeeded. `BLOCKED`, `FAIL`,
`STEP_FAILED` or `RUN_CRASHED` are not passing results. The GPU report and
resolved lock still require review before changing the project's environment
status or beginning Sprint 6. See [the campaign details](EXECUTION.md).

## 5. Optional: upload small evidence to GitHub

In cell 7 change `UPLOAD_TO_GITHUB = False` to `True`, then run it once.
The cell creates a new branch named `gpu/sprint5-<run-id>`, based on the exact
tested source commit, with evidence under `docs/gpu-runs/<run-id>/`.

It uploads selected JSON/text reports, package inventories, log excerpts and
artifact checksums. Model weights, adapter binaries, databases, environments and
caches remain in the ZIP. It does not edit `main`, automatically merge a PR,
change thresholds or mark `small-gpu-dev` ready. The token is used in request
headers only, not persisted in Git URLs, files or subprocess environments.

The cell prints the results-branch URL and records a local upload receipt. Share
that URL for review. If you do not upload, share the ZIP instead. Never share the
token. For HTTP 403, check the token's selected repository and Contents write
permission; do not print the token while diagnosing it.
