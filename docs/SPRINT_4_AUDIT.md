# Sprint 4 audit

Scope: roadmap version 1.1, Sprint 4; architecture Revision 3 sections 14 and 17.
Sprint 3 was validated before this work: 203 CPU tests passed, one slow test
deselected. Its existing implementation is preserved in a separate commit.

| Requirement | Implementation and evidence |
|---|---|
| Sandboxed math-verify pool | Pinned parser stack, isolated one-request subprocesses, bounded concurrency/input, hard wall deadlines, Unix CPU/memory limits; timeout and recovery golden cases |
| Stable checker identity | All mathematical dependency versions, extraction-policy version and configured deadline; incompatible packages fail before checks |
| Independent exact scorer | Standard-library-only import test retained; fixed and randomized exact-rational tests plus malformed-marker, exploit, format and truncation cases |
| Binary rewards | Correct/non-truncated = 1, otherwise 0; explicit optional overlong shaping; evaluation verdicts rejected |
| Pass-rate observations | Versioned Beta(1,1) inputs, canonical task binding, immutable raw completion/verdict artifacts, sample/group deduplication and checker-version filtering |
| Derived band and quotas | Measured-sample minimum, derived variance band, source/skill/low-entropy quotas, deterministic bounded selection and explicit pool exhaustion |
| Global exposure caps | Atomic slice/exposure transaction; retry, restart, concurrent-writer and failed-selection tests |
| Consensus flags | Two wrong-answer consensus estimates flag suspected labels; eligibility remains unchanged |
| Independent exclusion | Digest-bound, evidence-backed derivation/second-verifier/human attestations; reward checker cannot serve as its own second verifier |
| Resolution | Confirmed-good labels re-enter; corrected labels invalidate observations and remain retired with irreparable items; all evidence retained |
| Promotion audits | Durable edge-refresh plans and consecutive in-band audit flags; no exclusion by persistence |
| Trainer reachability | On-policy/offline gateways enforce measurement isolation and independent exclusions; previously reserved excluded items cannot be delivered |
| Checker disagreement | Paired correctness/status metrics isolated from reward production |
| Durable metadata | Additive migration 4, append-only observations/samples/reviews/slices/exposures/audit tables; historical science and job-result quarantine preserved |
| Public workflow | `training verify`, `sample`, `refresh`, `review`; pinned CPU CI, supplemental dependency lock, environment identities and usage guide |

## Validation

- Working-environment CPU suite: **240 passed, 1 slow test deselected**.
- Quick synthetic self-check: passed, including 4,000 randomized scorer cases,
  gate branches, bootstrap simulations and joint-power examples.
- Dependency consistency: `pip check` passed.
- Ruff: passed for the new modules and changed CLI/store/test code.
- Wheel build: passed with `--no-cache-dir --no-deps --no-build-isolation`;
  the initial attempt encountered the restricted global pip cache, so build
  caching was disabled.
- Clean-environment wheel acceptance: **241 passed, 1 slow test deselected**
  in 178.44 seconds, using a fresh CPython 3.13.7 environment and an installed
  wheel (no editable checkout on its import path).
- Final parser review additionally closed empty `Answer:` / final-phrase markers.
  The focused suite passed **34 tests, 1 slow test deselected** from source
  and again from the rebuilt final wheel (18.07 seconds). The final clean
  environment's dependency consistency check also passed.

## Scope and limits

The scorer fix is explicitly versioned as `exact_numeric@3`: malformed explicit
answer markers cannot be discarded alongside a valid marker. Old checker
identities and historical records are not rewritten. Statistical algorithms and
gate decisions are unchanged. There is no target-model or improvement claim.

Process isolation is for the constrained mathematical parser, not arbitrary
Python execution; Windows uses the parent deadline without Unix memory limits.
The golden suite exercises the supported extraction policy, not all mathematical
notation. A reviewer attests independence and the second checker's different
failure mode; software checks task/evidence binding and checker identity, not the
truth of an attestation. Test review records are explicitly synthetic and do not
approve the actual hidden-development pool.

Global caps count planned rollout-group reservations. A controller must reuse
the slice ID for retries and allocate a new ID for new work. Search exhaustion
fails conservatively. The single-source development fixtures need an explicit
fixture policy for sampling; production quotas are not silently relaxed.

Model generation, real training segments, target research hidden partitions and
scientific calibration remain outside Sprint 4. See [usage](VERIFICATION.md).
