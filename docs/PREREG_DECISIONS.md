# Phase I preregistration decisions

These choices must be supplied by the human before the draft can be validated or frozen.

1. **Practical gain floor.** Controls the minimum accuracy gain worth claiming and the sample size. Evaluation cost scales roughly as `1/floor²`. The architecture's synthetic example needs about 711 / 316 / 114 clusters per gate set for a 2 / 3 / 5 percentage-point floor; these are illustrations, and E1/E3 provide real estimates. What smallest gain (as a fraction) is worth claiming, and could E1 plausibly show it on the heavily post-trained 4B model?
2. **Four guardrail margins.** Control the maximum allowed loss versus base in transfer accuracy, instruction following, multilingual math, and format compliance. What accuracy fraction is the largest acceptable loss for each?
3. **Phase I compute budget.** Controls hardware fit and total experiment scope. The architecture's prior is a few GPU-days for E1+E2 on one 80 GB H100-class card at roughly 3k-token completions, and GPU-days to a few GPU-weeks for Phase I. What GPU model/count/VRAM, total GPU-hours, and calendar limit should be budgeted?
4. **Second E1a learning rate.** Controls the comparison alongside the fixed `2e-5` rate for two 200-step runs on `dev`. Which second rate should be used?
5. **GPU machine and lock target.** Controls whether the vLLM/Torch lock matches the Phase I host. What OS and GPU machine will run Phase I, and can dependency resolution be done on that machine?
6. **Prefilled statistical thresholds.** Controls false promotion and power: one-sided alpha `0.05` for both gates and guardrails; per-set/guardrail/joint power targets `0.90 / 0.97 / 0.80`; expected false halts at most `1.0`. Confirm these values or provide replacements.
