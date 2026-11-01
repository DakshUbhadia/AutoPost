"""autopost/eval/stats.py

Statistics for the measurement channel (Revision 2).

Changes from Revision 1 (see architecture doc, Appendix D, C1/C4):
  * The unit of resampling is the CLUSTER (dedup cluster; perturbation
    program for the hidden set), not the item. Items in one cluster are
    resampled together, so correlated outcomes are not counted as independent.
  * No inner resampling of the k samples per item. Resampling clusters with
    their samples attached already carries within-item sampling noise into
    the cluster totals; resampling the samples again counts it twice.
  * Candidate and reference may have different k (larger k for references,
    whose sampling error is shared across many comparisons).
  * Decisions use ONE-SIDED bounds at a pre-registered alpha.
  * Rates estimated by simulation are reported with Wilson intervals, so a
    result like "1 false positive in 50 runs" is shown as the wide interval
    it is.

Also contains the Phase I helpers for gate-power (E3), bootstrap-coverage
(E4) and shadow-controller (E5) simulations, which subsample stored
completions instead of generating new ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BootResult:
    mean: float          # point estimate of (candidate - reference)
    lower: float         # one-sided lower bound at level alpha
    upper: float         # one-sided upper bound at level alpha
    n_items: int
    n_clusters: int
    alpha: float


def _cluster_arrays(
    cand: Mapping[str, np.ndarray],
    ref: Mapping[str, np.ndarray],
    cluster_of: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray, int]:
    items = sorted(cand.keys() & ref.keys())
    if not items:
        raise ValueError("no shared items")
    # paired item-level difference of means; k may differ between models
    d = np.array([float(np.mean(cand[i])) - float(np.mean(ref[i])) for i in items])
    clusters = [cluster_of[i] for i in items]
    uniq = {c: j for j, c in enumerate(sorted(set(clusters)))}
    idx = np.array([uniq[c] for c in clusters])
    sums = np.bincount(idx, weights=d, minlength=len(uniq))
    counts = np.bincount(idx, minlength=len(uniq)).astype(float)
    return sums, counts, len(items)


def paired_cluster_bootstrap(
    cand: Mapping[str, np.ndarray],
    ref: Mapping[str, np.ndarray],
    cluster_of: Mapping[str, str],
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootResult:
    """Paired cluster bootstrap of the mean per-item difference (cand - ref).

    Resamples clusters with replacement; the statistic is the ratio
    sum(item differences) / number of items over the resampled clusters.
    Returns one-sided bounds: lower = alpha quantile, upper = 1-alpha quantile.
    """
    sums, counts, n_items = _cluster_arrays(cand, ref, cluster_of)
    C = len(sums)
    if C < 2:
        raise ValueError("need at least two clusters")
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, C, size=(n_boot, C))
    stats = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return BootResult(float(sums.sum() / counts.sum()), float(lo), float(hi),
                      n_items, C, alpha)


def cluster_se(
    cand: Mapping[str, np.ndarray],
    ref: Mapping[str, np.ndarray],
    cluster_of: Mapping[str, str],
) -> tuple[float, float, int]:
    """(mean difference, cluster-robust standard error, n_clusters).
    Closed-form companion to the bootstrap; used for sizing evaluation sets
    from pilot data (E1) and as a cross-check on bootstrap results."""
    sums, counts, _ = _cluster_arrays(cand, ref, cluster_of)
    C = len(sums)
    mean = sums.sum() / counts.sum()
    z = (sums - mean * counts) / counts.mean()      # ratio-estimator linearisation
    return float(mean), float(math.sqrt(z.var(ddof=1) / C)), C


def clusters_needed(se_at_c: float, c: int, effect: float, power_z: float,
                    alpha_z: float = 1.6449) -> int:
    """Clusters needed so that P(lower bound > 0 | true effect) reaches the
    target, given a pilot standard error se_at_c measured with c clusters
    (SE scales as 1/sqrt(C)). power_z: 0.8416 for 80%, 1.2816 for 90%."""
    if effect <= 0:
        raise ValueError("effect must be positive")
    return int(math.ceil(c * (se_at_c * (alpha_z + power_z) / effect) ** 2))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def subsample(
    stored: Mapping[str, np.ndarray],
    items: Sequence[str],
    k: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw k of the stored samples (without replacement) for each item.
    Emulates a fresh evaluation at sample size k from a large stored pool."""
    out = {}
    for i in items:
        s = stored[i]
        if len(s) < k:
            raise ValueError(f"item {i}: stored {len(s)} < k={k}")
        out[i] = s[rng.choice(len(s), size=k, replace=False)]
    return out


def sample_clusters(
    cluster_of: Mapping[str, str],
    n_clusters: int,
    rng: np.random.Generator,
) -> list[str]:
    """Pick n_clusters clusters without replacement; return all their items."""
    by_c: dict[str, list[str]] = {}
    for item, c in cluster_of.items():
        by_c.setdefault(c, []).append(item)
    keys = sorted(by_c)
    if n_clusters > len(keys):
        raise ValueError("not enough clusters stored")
    chosen = rng.choice(len(keys), size=n_clusters, replace=False)
    return [i for j in chosen for i in by_c[keys[j]]]


def simulate_rate(
    trial: Callable[[np.random.Generator], bool],
    n_sim: int,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Run a randomized yes/no trial n_sim times; return (rate, wilson_lo, wilson_hi).
    Used for: gate promotion probability at a given true effect (E3),
    A/A false-positive rate and interval coverage (E4), false halts (E5)."""
    rng = np.random.default_rng(seed)
    hits = sum(bool(trial(rng)) for _ in range(n_sim))
    lo, hi = wilson_interval(hits, n_sim)
    return hits / n_sim, lo, hi


def drop_saturated(
    base_passrate: Mapping[str, float],
    threshold: float,
) -> list[str]:
    """Stratification for gate sets: keep items whose base pass rate (estimated
    with samples NOT used in any comparison) is below `threshold`.
    Items every model solves carry almost no paired information."""
    return sorted(i for i, p in base_passrate.items() if p < threshold)


def scale_lora_state_dict(state: Mapping[str, object], alpha: float) -> dict:
    """E3 graded candidates: scale every LoRA B matrix by alpha.
    Since delta_W = (lora_alpha / r) * B @ A, this scales the learned update
    exactly by alpha (alpha=0 -> base, alpha=1 -> trained adapter)."""
    return {k: (v * alpha if "lora_B" in k else v) for k, v in state.items()}
