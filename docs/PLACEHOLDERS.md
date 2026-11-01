# Configuration placeholders

The following unresolved values in `configs/base.yaml` are intentional and owned by the named experiment or preregistration.

| Placeholder path(s) | Owner |
|---|---|
| `model_profiles.*.revision`, `train.trl_version`, `data.dedup.embed_model_revision` | T4/T5 verified pins |
| `model_profiles.*.rollout_sampling.max_new_tokens`, `model_profiles.*.eval_sampling.max_new_tokens` | E1 (provisional 16,384 for the spike) |
| `model_profiles.*.max_prompt_tokens` | Spike / E1 |
| `data.sizes_in_clusters.*` | E1 + E3 (hidden sets: program-level) |
| `verifier_pool.workers` | E1 (GPU idle time waiting on verification) |
| `train.trl.vllm_gpu_memory_utilization` | E1 |
| `eval.k.*` | E3 / E4 |
| `gate.floor`, `gate.guardrails[*].margin` | Phase I preregistration (T11) |
| `controller.gate_every_steps`, `controller.step_budget` | E1 + E3, then Phase II preregistration |
| `controller.read_budgets.{gate_a,gate_b,hidden_audit}`, `controller.human_change_budget` | Phase II preregistration |
| `tripwires.*.alpha`, `tripwires.pool_inband_min_frac` | Phase II (allocated from false-halt budget; E5 checks) |
| `tripwires.drift_limits.*` | E1 (absolute limits) |
