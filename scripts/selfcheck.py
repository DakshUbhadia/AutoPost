"""selfcheck.py -- validates the Revision 2 companion modules.

Run:  python selfcheck.py
Synthetic-data demonstrations only; they validate the statistics code and
illustrate the review findings. They do NOT replace E3/E4 on real model
outputs, which are required by Level 0 (architecture doc §36).
"""
from __future__ import annotations

import ast
import argparse
from pathlib import Path
import math
import random
from fractions import Fraction

import numpy as np

from autopost.verify import exact_scorer as xs
from autopost.interfaces import (BinaryReward, Completion, DefaultGate, GateSpec, GuardrailSpec,
                                 Outcome, Task, Verdict, band_for)
from autopost.eval.stats import (BootResult, cluster_se, clusters_needed, paired_cluster_bootstrap,
                                 scale_lora_state_dict, simulate_rate, wilson_interval)


def section(t):
    print(f"\n== {t}")


# ------------------------------------------------------------------ scorer
def scorer_tests(n_random: int = 4000) -> None:
    section("exact scorer: fixed cases")
    cases = [
        ("so the answer is \\boxed{42}", "42", True, "ok"),
        ("\\boxed{\\frac{3}{4}}", "0.75", True, "ok"),
        ("\\boxed{\\dfrac{-3}{4}}", "-3/4", True, "ok"),
        ("\\boxed{-\\frac{3}{4}}", "-3/4", True, "ok"),
        ("\\boxed{\\frac34}", "3/4", True, "ok"),
        ("Answer: 1,250", "1250", True, "ok"),
        ("**Final Answer:** 7", "7", True, "ok"),
        ("\\boxed{x = 5}", "5", True, "ok"),
        ("\\boxed{12\\text{ apples}}", "12", True, "ok"),
        ("\\boxed{25\\%}", "25", True, "ok"),
        ("The final answer is 9.", "9", True, "ok"),
        ("We get 3 * 4 = 12\nSo there are 12 eggs", "12", True, "ok"),
        ("\\boxed{2} or maybe \\boxed{3}", "2", False, "multiple_answers"),
        ("\\boxed{2}\nAnswer: 3", "2", False, "multiple_answers"),
        ("It is 2 or 3", "2", False, "multiple_answers"),
        ("\\boxed{0.5}\nAnswer: 1/2", "1/2", True, "ok"),          # same value, two markers
        ("\\boxed{}", "1", False, "parse_error"),
        ("\\boxed{\\sqrt{2}}", "1", False, "parse_error"),
        ("no numbers here", "1", False, "no_answer"),
        ("", "1", False, "no_answer"),
        ("\\boxed{1/0}", "1", False, "parse_error"),
    ]
    bad = 0
    for text, ref, want_ok, want_status in cases:
        v = xs.score_content(text, ref)
        if v.correct != want_ok or v.status != want_status:
            bad += 1
            print("  FAIL", repr(text), ref, v)
    assert xs.score_content("\\boxed{42}", "42", truncated=True).status == "truncated"
    print(f"  {len(cases) - bad}/{len(cases)} fixed cases pass")
    assert bad == 0

    section(f"exact scorer: {n_random} randomized equivalence cases")
    rng = random.Random(7)

    def renders(fr: Fraction) -> list[str]:
        p, q = fr.numerator, fr.denominator
        out = [f"\\boxed{{{p}/{q}}}", f"\\boxed{{\\frac{{{p}}}{{{q}}}}}",
               f"\\boxed{{\\dfrac{{{p}}}{{{q}}}}}", f"Answer: {p}/{q}"]
        if q == 1:
            out += [f"\\boxed{{{p}}}", f"Answer: {p:,}", f"The final answer is {p}."]
        qq = q
        for f in (2, 5):
            while qq % f == 0:
                qq //= f
        if qq == 1:                                  # terminating decimal
            out.append(f"\\boxed{{{str(float(fr)) if abs(p) < 10**12 else ''}}}")
        return [r for r in out if r != "\\boxed{}"]

    fails = 0
    for _ in range(n_random):
        q = rng.choice([1, 1, 1, 2, 3, 4, 5, 8, 10, 16, 25, 7, 12])
        p = rng.randint(-10**6, 10**6)
        fr = Fraction(p, q)
        ref = f"{fr.numerator}/{fr.denominator}" if fr.denominator != 1 else str(fr.numerator)
        for r in renders(fr):
            if not xs.score_content(r, ref).correct:
                fails += 1
        wrong = fr + Fraction(1, fr.denominator)
        if xs.score_content(f"\\boxed{{{wrong.numerator}/{wrong.denominator}}}", ref).correct:
            fails += 1
        if xs.score_content(f"\\boxed{{{p}/{q}}} or \\boxed{{{p + 1}/{q}}}", ref).status != "multiple_answers":
            fails += 1
    print(f"  failures: {fails}")
    assert fails == 0

    section("format compliance")
    assert xs.format_compliant("work...\n\\boxed{3}", "boxed")
    assert not xs.format_compliant("work...\nAnswer: 3", "boxed")
    assert xs.format_compliant("work...\nAnswer: 3", "answer_line")
    assert not xs.format_compliant("Answer: 3\nmore text", "answer_line")
    assert not xs.format_compliant("\\boxed{3} and \\boxed{4}", "boxed")
    print("  ok")


def independence_test() -> None:
    section("independence: exact_scorer imports")
    tree = ast.parse(Path(xs.__file__).read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    forbidden = {"math_verify", "sympy", "latex2sympy2", "latex2sympy2_extended"}
    print("  imports:", sorted(mods))
    assert not (mods & forbidden) and not any("train" in m for m in mods)


# ------------------------------------------------------------- reward/band
def reward_and_band() -> None:
    section("reward and band")
    t = Task("t", "c", "s", "k", "p", "1", "int", False, "train")
    c = lambda n, tr=False: Completion("x", "t", "s", "T", 0, 0, "", n, tr)
    ok = Verdict(True, "ok", "1", "v")
    r = BinaryReward(4096)
    assert r(t, c(100), ok).value == 1.0 and r(t, c(4096, True), ok).value == 0.0
    print("  truncated -> 0 (not -1); band G=8:", tuple(round(x, 3) for x in band_for(8, 0.5)))


# --------------------------------------------------------------- C1 demo
def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def c1_demo() -> None:
    section("C1: margin rule vs test-against-zero (normal approximation)")
    z2, zb = 1.96, 0.8416
    mde = z2 + zb                                    # in units of sigma
    p_old_at_mde = 1 - Phi((mde + z2) - mde)         # need est > mde + 1.96
    true_needed = mde + z2 + zb
    print(f"  Rev.1: MDE = {mde:.2f} sigma; P(pass | true gain = MDE) = {p_old_at_mde:.3f}; "
          f"true gain for 80% pass = {true_needed:.2f} sigma")
    z1 = 1.645                                       # one-sided alpha 0.05
    # Rev.2: size each of Gate-A and Gate-B so that P(lower>0 | floor) = 0.9 -> joint ~0.81
    s = (z1 + 1.2816)                                # floor = s * sigma_per_gate
    p_each = 1 - Phi(z1 - s)
    print(f"  Rev.2: sets sized so floor = {s:.2f} sigma each; P(pass A) = P(pass B) = {p_each:.2f}; "
          f"joint = {p_each**2:.2f}; joint false promotion under null = {0.05**2:.4f}")


# ------------------------------------------------------- data simulator
def make_population(C, m, rng, mu_logit=0.0, tau=0.0, sd_cluster=1.0, sd_item=0.5):
    a = rng.normal(0, sd_cluster, C)
    delta = rng.normal(mu_logit, tau, C) if tau > 0 else np.full(C, mu_logit)
    items, cluster_of, p_ref, p_cand = [], {}, {}, {}
    for c in range(C):
        for j in range(m):
            iid = f"c{c}_i{j}"
            e = rng.normal(0, sd_item)
            lr = a[c] + e
            items.append(iid)
            cluster_of[iid] = f"c{c}"
            p_ref[iid] = 1 / (1 + math.exp(-lr))
            p_cand[iid] = 1 / (1 + math.exp(-(lr + delta[c])))
    return items, cluster_of, p_ref, p_cand


def draw(p, items, k, rng):
    return {i: (rng.random(k) < p[i]).astype(float) for i in items}


def rev1_hierarchical(cand, ref, n_boot=1000, alpha_one_sided=0.05, seed=0):
    items = sorted(cand)
    A = np.stack([cand[i] for i in items]); B = np.stack([ref[i] for i in items])
    n, k = A.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    sa = rng.integers(0, k, size=(n_boot, n, k)); sb = rng.integers(0, k, size=(n_boot, n, k))
    a = np.take_along_axis(A[idx], sa, axis=2).mean(axis=(1, 2))
    b = np.take_along_axis(B[idx], sb, axis=2).mean(axis=(1, 2))
    lo, hi = np.quantile(a - b, [alpha_one_sided, 1 - alpha_one_sided])
    return lo, hi


def item_bootstrap(cand, ref, n_boot=1000, alpha=0.05, seed=0):
    items = sorted(cand)
    d = np.array([cand[i].mean() - ref[i].mean() for i in items])
    rng = np.random.default_rng(seed)
    s = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return np.quantile(s, [alpha, 1 - alpha])


def c4_demo(n_sim=1500) -> None:
    section(f"C4-a: A/A (no true difference), one-sided alpha = 0.05, 80 clusters x 3 items, k=4, {n_sim} runs")
    fp = {"rev1_hier": 0, "cluster": 0}
    rng = np.random.default_rng(1)
    for s in range(n_sim):
        items, cof, pr, pc = make_population(80, 3, rng)
        A, B = draw(pc, items, 4, rng), draw(pr, items, 4, rng)
        lo, _ = rev1_hierarchical(A, B, seed=s)
        fp["rev1_hier"] += lo > 0
        fp.setdefault("cluster_se", 0)
        m_, se_, _ = cluster_se(A, B, cof)
        fp["cluster_se"] += (m_ - 1.6449 * se_) > 0
        fp["cluster"] += paired_cluster_bootstrap(A, B, cof, n_boot=1000, seed=s).lower > 0
    for k_, v in fp.items():
        lo, hi = wilson_interval(v, n_sim)
        print(f"  {k_:10s} false-positive rate {v/n_sim:.3f}  (Wilson 95%: {lo:.3f}-{hi:.3f})")

    section(f"C4-b: effect varies by program (tau > 0): coverage of the population mean gain, {n_sim} runs")
    rng = np.random.default_rng(2)
    big_items, _, bpr, bpc = make_population(20000, 1, np.random.default_rng(99), mu_logit=0.3, tau=0.8)
    mu = float(np.mean([bpc[i] - bpr[i] for i in big_items]))
    cov = {"item_level": 0, "rev1_hier": 0, "cluster": 0}
    for s in range(n_sim):
        items, cof, pr, pc = make_population(60, 8, rng, mu_logit=0.3, tau=0.8)
        A, B = draw(pc, items, 4, rng), draw(pr, items, 4, rng)
        cov["item_level"] += item_bootstrap(A, B, seed=s)[0] <= mu
        cov["rev1_hier"] += rev1_hierarchical(A, B, seed=s)[0] <= mu
        cov["cluster"] += paired_cluster_bootstrap(A, B, cof, n_boot=1000, seed=s).lower <= mu
    print(f"  population mean gain = {mu:.4f}; nominal one-sided coverage 0.95")
    for k_, v in cov.items():
        lo, hi = wilson_interval(v, n_sim)
        print(f"  {k_:10s} coverage {v/n_sim:.3f}  (Wilson 95%: {lo:.3f}-{hi:.3f})")


def joint_power_demo(n_sim=400) -> None:
    section("Joint gate power (Gate-A AND Gate-B, fresh reference samples), synthetic")
    def trial_factory(mu_logit):
        def trial(rng):
            for _split in ("a", "b"):
                items, cof, pr, pc = make_population(150, 2, rng, mu_logit=mu_logit)
                A, B = draw(pc, items, 4, rng), draw(pr, items, 8, rng)   # k_ref = 2k
                if paired_cluster_bootstrap(A, B, cof, n_boot=800,
                                            seed=int(rng.integers(1e9))).lower <= 0:
                    return False
            return True
        return trial
    print("  each gate set: 150 clusters x 2 items, k=4 candidate, k_ref=8, one-sided alpha 0.05")
    for mu in (0.0, 0.15, 0.3, 0.45):
        bi, _, bpr, bpc = make_population(20000, 1, np.random.default_rng(3), mu_logit=mu)
        gain = float(np.mean([bpc[i] - bpr[i] for i in bi]))
        r, lo, hi = simulate_rate(trial_factory(mu), n_sim, seed=int(mu * 100))
        print(f"  true accuracy gain {gain:+.3f}: P(promote) = {r:.3f}  (Wilson 95%: {lo:.3f}-{hi:.3f})")


# --------------------------------------------------------------- gate
class FakeCtx:
    def __init__(self, vals, steps=0, budgets=None, fired=()):
        self.vals, self.steps_since_promotion = vals, steps
        self.budgets = budgets or {"gate_a": 5, "gate_b": 5}
        self.fired = list(fired)
    def primary(self, split, c, r):
        m, lo, hi = self.vals.get(split, (0.04, 0.01, 0.07))
        return BootResult(m, lo, hi, 300, 150, 0.05)
    def guardrail(self, g, c, b):
        m, lo, hi = self.vals.get(g.name, (0.0, -0.01, 0.01))
        return BootResult(m, lo, hi, 300, 150, g.alpha)
    def tripwires_fired(self): return self.fired
    def budget_left(self, b): return self.budgets.get(b, 1)


def gate_tests() -> None:
    section("gate branches")
    spec = GateSpec(0.05, 0.05, (GuardrailSpec("transfer", 0.02, 0.05),
                                 GuardrailSpec("format_compliance", 0.03, 0.05)), step_budget=1000)
    g = DefaultGate(spec)
    cases = [
        ("all good", FakeCtx({}), Outcome.PROMOTE),
        ("gate_a inconclusive", FakeCtx({"gate_a": (0.01, -0.01, 0.03)}), Outcome.CONTINUE),
        ("gate_a inconclusive, budget spent", FakeCtx({"gate_a": (0.01, -0.01, 0.03)}, steps=1000), Outcome.HALT),
        ("gate_a confident regression", FakeCtx({"gate_a": (-0.03, -0.05, -0.01)}), Outcome.RESTORE),
        ("guardrail not demonstrated", FakeCtx({"transfer": (-0.01, -0.03, 0.01)}), Outcome.CONTINUE),
        ("guardrail violated", FakeCtx({"transfer": (-0.05, -0.07, -0.03)}), Outcome.RESTORE),
        ("gate_b not confirmed", FakeCtx({"gate_b": (0.01, -0.01, 0.03)}), Outcome.CONTINUE),
        ("gate_b budget spent", FakeCtx({}, budgets={"gate_a": 3, "gate_b": 0}), Outcome.HALT),
        ("tripwire", FakeCtx({}, fired=["checker_disagreement"]), Outcome.HALT),
    ]
    for name, ctx, want in cases:
        got = g.decide(ctx, "cand", "champ", "base").outcome
        print(f"  {name:36s} -> {got.name:8s} {'ok' if got == want else 'FAIL'}")
        assert got == want


def lora_scale_test() -> None:
    section("LoRA scaling for graded candidates (E3)")
    sd = {"layer.lora_A.weight": np.ones((2, 4)), "layer.lora_B.weight": np.full((4, 2), 3.0)}
    out = scale_lora_state_dict(sd, 0.25)
    assert np.allclose(out["layer.lora_B.weight"], 0.75) and np.allclose(out["layer.lora_A.weight"], 1.0)
    print("  B scaled, A unchanged: ok")


def sizing_demo() -> None:
    section("Sizing from a pilot: clusters needed for the floor at 90% power per gate set")
    rng = np.random.default_rng(5)
    items, cof, pr, pc = make_population(200, 2, rng, mu_logit=0.3)
    A, B = draw(pc, items, 4, rng), draw(pr, items, 8, rng)
    m, se, C = cluster_se(A, B, cof)
    for floor in (0.02, 0.03, 0.05):
        print(f"  pilot SE {se:.4f} at {C} clusters -> floor {floor:.2f}: "
              f"{clusters_needed(se, C, floor, power_z=1.2816)} clusters")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    scorer_tests()
    independence_test()
    reward_and_band()
    gate_tests()
    lora_scale_test()
    c1_demo()
    c4_demo(n_sim=200 if args.quick else 1500)
    joint_power_demo(n_sim=50 if args.quick else 400)
    sizing_demo()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
