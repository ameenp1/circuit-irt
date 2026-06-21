"""Reference-solution generation + harness verification (Week 3).

For each candidate item from the spec-template generators we build a real
reference circuit, run it through the harness, and KEEP the item iff its
reference verifies. Items are **reference-grounded**: targets come from the
reference's measured metrics (slack shrinks with tightness), so every kept item
provably has a passing solution. Discards happen only on real failure
(non-convergence / unmeasurable required metric).

  fix   : an optional objective that comes back NaN is dropped from the item.
  discard: non-convergence, or a REQUIRED metric is NaN/unmeasurable.

References: filters = passive 2nd-order RLC (Butterworth); CS amp / diff pair =
level-1 MOSFET; two-stage op-amp = behavioral (A0/GBW/PM placed analytically).
Output: candidate_bank.json (verified items + reference netlists + metrics).
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter

from circuit_irt.families import Direction, FAMILIES
from circuit_irt.harness import (AnalysisPlan, ItemSpec, _to_lin,
                     simulate, extract_metrics, score, classify, fingerprint)
from circuit_irt.generators import generate_items, mark_reliability_subset
from circuit_irt.paths import DATA

NMOS = ".model NM NMOS (LEVEL=1 VTO=0.45 KP=120u LAMBDA=0.02)"

ANALYSIS_FOR = {
    "gain_db": "ac", "dc_gain_db": "ac", "diff_gain_db": "ac",
    "passband_gain_db": "ac", "bw_3db_hz": "ac", "cutoff_freq_hz": "ac",
    "stopband_atten_db": "ac", "gbw_hz": "ac", "phase_margin_deg": "ac",
    "q_factor": "ac", "quiescent_power_w": "op", "out_swing_vpp": "dc",
    "cmrr_db": "ac_cmrr", "icmr_v": "dc_icmr",
}

# per-family candidate counts — balanced across families; the generator's tier
# weights (15/20/30/35) keep the hard-tail surplus inside each family.
COUNTS = {"filters": 125, "cs_amp": 125, "diff_pair": 125, "two_stage_opamp": 125}

# objectives each family's reference can actually measure. Unsupported optionals
# are dropped from an item (a "fix"); q_factor (band-pass) isn't realized by the
# Butterworth LP reference, and the behavioral op-amp supports AC metrics only
# (power/swing/slew need a transistor design — future work).
MEASURABLE = {
    "filters": {"cutoff_freq_hz", "passband_gain_db", "stopband_atten_db"},
    "cs_amp": {"gain_db", "bw_3db_hz", "out_swing_vpp", "quiescent_power_w"},
    "diff_pair": {"diff_gain_db", "cmrr_db", "icmr_v", "quiescent_power_w", "bw_3db_hz"},
    "two_stage_opamp": {"dc_gain_db", "gbw_hz", "phase_margin_deg"},
}


# --------------------------------------------------------------------------- #
# reference circuits per family
# --------------------------------------------------------------------------- #
def design_filter(item, rng) -> str:
    fc = item.targets["cutoff_freq_hz"]            # honor the sampled cutoff
    w = 2 * math.pi * fc
    C = 100e-9
    L = 1.0 / (w * w * C)
    R = math.sqrt(L / C) / 0.707                    # Butterworth (maximally flat)
    return f"R1 in n1 {R:.6g}\nL1 n1 out {L:.6g}\nC1 out 0 {C:.6g}"


def design_cs(item, rng) -> str:
    w = rng.uniform(60, 200)
    rd = rng.uniform(1.2e3, 3.0e3)
    return f"{NMOS}\nM1 out in 0 0 NM W={w:.0f}u L=1u\nRD vdd out {rd:.0f}"


def design_diff(item, rng) -> str:
    w = rng.uniform(150, 240)
    rd = rng.uniform(1.6e3, 2.6e3)
    vnb = rng.uniform(0.66, 0.76)
    return (f"{NMOS}\n"
            f"M1 outp inp tail 0 NM W={w:.0f}u L=1u\nM2 out inm tail 0 NM W={w:.0f}u L=1u\n"
            f"RD1 vdd outp {rd:.0f}\nRD2 vdd out {rd:.0f}\n"
            f"M3 tail nb 0 0 NM W=200u L=1u\nVnb nb 0 {vnb:.3f}")


def design_opamp(item, rng) -> str:
    d = item.difficulty_score
    a0_db = 70 + 20 * d
    gbw = 10 ** (6 + 1.5 * d)
    pm = 70 - 12 * d
    a0 = 10 ** (a0_db / 20)
    p1 = gbw / a0
    c1 = 1 / (2 * math.pi * p1 * 1e3)
    p2 = gbw * math.tan(math.radians(pm))
    c2 = 1 / (2 * math.pi * p2 * 1e3)
    return (f"E1 n1 0 inp inm {a0:.6g}\nR1 n1 n2 1k\nC1 n2 0 {c1:.6e}\n"
            f"E2 n3 0 n2 0 1\nR2 n3 out 1k\nC2 out 0 {c2:.6e}")


DESIGNERS = {"filters": design_filter, "cs_amp": design_cs,
             "diff_pair": design_diff, "two_stage_opamp": design_opamp}


# --------------------------------------------------------------------------- #
# per-item analysis plan
# --------------------------------------------------------------------------- #
def make_plan(item, analyses) -> AnalysisPlan:
    fid = item.family_id
    if fid == "filters":
        fc = item.targets["cutoff_freq_hz"]
        return AnalysisPlan(analyses, in_nodes=("in",), ac=(1, 1e8, "dec", 40),
                            f_ref=max(1.0, fc / 100))
    if fid == "cs_amp":
        return AnalysisPlan(analyses, in_nodes=("in",), input_bias=0.78,
                            supplies=(("VDD", "vdd", 1.8),), ac=(1, 1e9, "dec", 30),
                            f_ref=1e3, dc_sweep=(0.0, 1.5, 0.02), load_cap=1e-12)
    if fid == "diff_pair":
        return AnalysisPlan(analyses, in_nodes=("inp", "inm"), input_mode="differential",
                            input_bias=0.9, out_node="out", supplies=(("VDD", "vdd", 1.8),),
                            ac=(1, 5e8, "dec", 30), f_ref=1e3, cmrr_fref=1e3,
                            icmr=(0.2, 1.7, 0.15), load_cap=5e-12)
    return AnalysisPlan(analyses, in_nodes=("inp", "inm"), input_mode="differential",
                        ac=(1, 1e9, "dec", 40), f_ref=10)


# --------------------------------------------------------------------------- #
# reference-grounded spec (slack shrinks with tightness)
# --------------------------------------------------------------------------- #
def ground_spec(family, metrics, objectives, tightness):
    slack = 0.30 / tightness                         # 0.30 (t=1) .. 0.0375 (t=8)
    targets, tol = {}, {}
    for k in objectives:
        ms = family.metric(k); meas = metrics[k]
        if ms.direction is Direction.GE:
            lin = _to_lin(meas, ms.unit) * (1 - slack)
            targets[k] = 20 * math.log10(lin) if ms.unit == "dB" else lin
        elif ms.direction is Direction.LE:
            targets[k] = meas * (1 + slack)
        elif ms.direction is Direction.TARGET:
            targets[k] = meas
            tol[k] = ms.base_tol / tightness
        else:                                        # WINDOW
            targets[k] = ms.sample_range
    return targets, tol


# --------------------------------------------------------------------------- #
# build + verify
# --------------------------------------------------------------------------- #
def build_bank(seed: int = 7):
    rng = random.Random(seed)
    candidates = []
    for fid, n in COUNTS.items():
        candidates += generate_items(FAMILIES[fid], n, rng)

    kept, discards, fixes = [], Counter(), Counter()
    for it in candidates:
        fam = FAMILIES[it.family_id]
        objs = [k for k in it.objectives if k in MEASURABLE[it.family_id]]
        if len(objs) < len(it.objectives):
            fixes[it.family_id] += 1
        analyses = tuple(sorted({ANALYSIS_FOR[k] for k in objs}))
        netlist = DESIGNERS[it.family_id](it, rng)
        raw = simulate(netlist, make_plan(it, analyses))
        if not raw.converged:
            discards[f"{it.family_id}:nonconvergence"] += 1
            continue
        metrics = extract_metrics(raw, fam)

        # fix (drop NaN optional) / discard (NaN required)
        bad_required = False
        active = []
        for k in objs:
            v = metrics.get(k, float("nan"))
            if isinstance(v, float) and math.isnan(v):
                if fam.metric(k).required:
                    bad_required = True
                    discards[f"{it.family_id}:nan_required:{k}"] += 1
                    break
                # else: drop optional objective (fix)
            else:
                active.append(k)
        if bad_required or not active:
            if not bad_required:
                discards[f"{it.family_id}:no_active_objectives"] += 1
            continue

        targets, tol = ground_spec(fam, metrics, active, it.tightness)
        spec = ItemSpec(fam, targets, tuple(active), tol, it.corner)
        s = score(metrics, spec, syntax_ok=raw.syntax_ok, converged=raw.converged)
        label = classify(netlist, raw, s, fingerprint(netlist))
        if not s["all_pass"]:                        # reference must pass its own spec
            discards[f"{it.family_id}:reference_fails_grounded_spec"] += 1
            continue

        kept.append(dict(
            item_id=it.item_id, family_id=it.family_id, tier=it.tier,
            tightness=it.tightness, corner=it.corner,
            objectives=list(active), n_objectives=len(active),
            targets={k: (list(v) if isinstance(v, tuple) else v) for k, v in targets.items()},
            tolerance=tol, difficulty_score=it.difficulty_score,
            reference_netlist=netlist,
            reference_metrics={k: round(float(metrics[k]), 6) for k in active},
            reference_label=label.value,
        ))
    return kept, discards, fixes


if __name__ == "__main__":
    kept, discards, fixes = build_bank()
    # mark reliability subset on the verified bank (~50 spanning family×tier)
    by_cell = {}
    rng = random.Random(11)
    rel_ids = set()
    for fid in FAMILIES:
        for tier in ("easy", "medium", "hard", "extreme"):
            cell = [r for r in kept if r["family_id"] == fid and r["tier"] == tier]
            for r in rng.sample(cell, min(3, len(cell))):
                rel_ids.add(r["item_id"])
    for r in kept:
        r["reliability_subset"] = r["item_id"] in rel_ids

    n_cand = sum(COUNTS.values())
    fam_mix = Counter(r["family_id"] for r in kept)
    tier_mix = Counter(r["tier"] for r in kept)
    print(f"candidates: {n_cand}  ->  kept: {len(kept)}  "
          f"discarded: {sum(discards.values())} ({sum(discards.values())/n_cand:.0%})")
    print("kept by family:", dict(fam_mix))
    print("kept by tier:  ", dict(tier_mix),
          f"-> hard-tail {100*(tier_mix['hard']+tier_mix['extreme'])/len(kept):.0f}%")
    print("discard reasons:", dict(discards) or "none")
    print("items fixed (unmeasurable optional dropped):", dict(fixes) or "none")
    print("avg objectives/item:",
          round(sum(r["n_objectives"] for r in kept) / len(kept), 2))
    print(f"reliability subset: {len(rel_ids)} items")
    assert all(r["reference_label"] == "pass" for r in kept), "a reference didn't pass"

    with open(DATA / "candidate_bank.json", "w") as f:
        json.dump({"items": kept, "n_candidates": n_cand,
                   "discards": dict(discards)}, f, indent=1)
    print(f"\nwrote candidate_bank.json ({len(kept)} verified items)")

    assert len(kept) >= 200, f"need >=200 verified items, got {len(kept)}"
    assert len(fam_mix) == 4, "every family represented"
    # spot-check: every kept item's stored reference actually re-verifies
    import random as _r
    for r in _r.Random(0).sample(kept, 12):
        fam = FAMILIES[r["family_id"]]
        analyses = tuple(sorted({ANALYSIS_FOR[k] for k in r["objectives"]}))
        it = type("X", (), {"family_id": r["family_id"], "targets": r["targets"]})
        raw = simulate(r["reference_netlist"], make_plan(it, analyses))
        m = extract_metrics(raw, fam)
        tg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in r["targets"].items()}
        spec = ItemSpec(fam, tg, tuple(r["objectives"]), r["tolerance"], r["corner"])
        assert score(m, spec, syntax_ok=raw.syntax_ok, converged=raw.converged)["all_pass"], \
            f"{r['item_id']} re-verify failed"
    print("OK: bank built; sampled references re-verify through the harness.")
