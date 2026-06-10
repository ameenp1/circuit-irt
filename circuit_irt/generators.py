"""Spec-template generators + paraphrase generator (Week 3).

generate_bank(...) samples concrete items from each FamilySpec across the three
difficulty axes (constraint-tightness, # simultaneous objectives, robustness
corner), deliberately **over-sampling the hard tail** — surplus hard items are
pruned after calibration (Week 8), not patched after freeze.

Each item carries 2-3 paraphrase surface forms (identical constraints, different
wording / ordering / units phrasing) so Week 7 can measure test-retest reliability
of θ. mark_reliability_subset(...) flags ~50 items spanning families × difficulty.

An item's structured spec -> harness.ItemSpec via .to_item_spec(); its prompt
forms via .forms(). Reference-solution verification is the NEXT task.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from circuit_irt.families import Direction, FamilySpec, FAMILIES
from circuit_irt.harness import ItemSpec, _to_lin


def spec_from_metrics(family: FamilySpec, metrics: dict, objectives,
                      *, ge: float = 0.7, le: float = 1.5,
                      target_tol: float | None = None) -> ItemSpec:
    """Build a scorable ItemSpec whose targets are derived from measured metrics
    (GE floor below measured, LE ceiling above, TARGET at measured ± tol,
    WINDOW = family band). Used to verify reference solutions pass their own spec
    and to construct the memorization probes."""
    targets, tol = {}, {}
    for k in objectives:
        ms = family.metric(k); meas = metrics[k]
        if ms.direction is Direction.GE:
            lin = _to_lin(meas, ms.unit) * ge
            targets[k] = 20 * math.log10(lin) if ms.unit == "dB" else lin
        elif ms.direction is Direction.LE:
            targets[k] = meas * le
        elif ms.direction is Direction.TARGET:
            targets[k] = meas
            tol[k] = ms.base_tol if target_tol is None else target_tol
        else:                                          # WINDOW
            targets[k] = ms.sample_range
    return ItemSpec(family, targets, tuple(objectives), tol)

# metrics whose ngspice recipe isn't implemented yet -> keep them out of the
# sampled objectives so every generated item stays verifiable (slew: TBD W3).
UNIMPLEMENTED = {"slew_rate_vps"}

# --------------------------------------------------------------------------- #
# difficulty tiers — hard tail deliberately over-weighted (65% hard+extreme vs
# 50% uniform). Each tier sets the tightness levels, how many OPTIONAL objectives
# to stack on top of the always-on required metrics, and the corner probability.
# --------------------------------------------------------------------------- #
TIERS = ("easy", "medium", "hard", "extreme")
TIER_WEIGHTS = (0.15, 0.20, 0.30, 0.35)

TIER_CFG = {
    "easy":    dict(tight=(1.0,),       n_opt=(0, 0), p_corner=0.0),
    "medium":  dict(tight=(1.0, 2.0),   n_opt=(0, 1), p_corner=0.2),
    "hard":    dict(tight=(2.0, 4.0),   n_opt=(1, 3), p_corner=0.5),
    "extreme": dict(tight=(4.0, 8.0),   n_opt=(2, 9), p_corner=0.9),
}

CORNER_DESC = {
    "comp_tol_5pct": "±5% component tolerance",
    "comp_tol_10pct": "±10% component tolerance",
    "supply_pm10": "±10% supply variation",
    "temp_0_85C": "the 0–85 °C temperature range",
    "vth_corner": "the slow/fast Vth process corners",
    "tail_finite_ro": "a non-ideal (finite-rₒ) tail current source",
    "mismatch_1pct": "1% device mismatch",
    "cload_1_10pf": "a 1–10 pF load capacitance",
}


@dataclass
class GeneratedItem:
    item_id: str
    family_id: str
    tier: str
    tightness: float
    corner: str
    objectives: tuple[str, ...]
    targets: dict                       # key -> float | (lo, hi)
    tolerance: dict                     # key -> frac tol (TARGET metrics)
    n_objectives: int
    difficulty_score: float
    reliability_subset: bool = False

    def to_item_spec(self) -> ItemSpec:
        return ItemSpec(FAMILIES[self.family_id], self.targets,
                        self.objectives, self.tolerance, self.corner)

    def forms(self) -> list[str]:
        fam = FAMILIES[self.family_id]
        return [render_prompt(self, fam, f) for f in range(3)]


# --------------------------------------------------------------------------- #
# target sampling
# --------------------------------------------------------------------------- #
def _interp(lo, hi, frac, unit):
    """Position a target in [lo, hi]; geometric for wide multiplicative ranges."""
    frac = min(1.0, max(0.0, frac))
    if unit in ("Hz", "W") and lo > 0 and hi > 0:
        return lo * (hi / lo) ** frac
    return lo + (hi - lo) * frac


def _sample_target(metric, tightness, rng):
    """Sample a target + tolerance; tightness pushes toward the hard end."""
    lo, hi = metric.sample_range
    d = math.log2(tightness) / math.log2(8.0)          # 0 (t=1) .. 1 (t=8)
    jit = rng.uniform(-0.08, 0.08)
    tol = metric.base_tol
    if metric.direction is Direction.GE:               # higher target = harder
        return _interp(lo, hi, d + jit, metric.unit), tol
    if metric.direction is Direction.LE:               # lower ceiling = harder
        return _interp(lo, hi, 1.0 - d + jit, metric.unit), tol
    if metric.direction is Direction.TARGET:           # tighter tol = harder
        return _interp(lo, hi, rng.uniform(0, 1), metric.unit), tol / tightness
    return (lo, hi), tol                               # WINDOW: the band itself


# --------------------------------------------------------------------------- #
# item generation
# --------------------------------------------------------------------------- #
def _gen_item(family: FamilySpec, tier: str, idx: int, rng: random.Random) -> GeneratedItem:
    cfg = TIER_CFG[tier]
    d = family.difficulty
    pool = [k for k in d.objective_pool if k not in UNIMPLEMENTED]
    required = [m.key for m in family.metrics if m.required and m.key in pool]
    optional = [k for k in pool if k not in required]

    tight = rng.choice(cfg["tight"])
    headroom = max(0, d.max_objectives - len(required))
    n_opt = min(rng.randint(*cfg["n_opt"]), len(optional), headroom)
    objectives = tuple(required + rng.sample(optional, n_opt))

    corner = ("none" if rng.random() > cfg["p_corner"]
              else rng.choice([c for c in d.corner_options if c != "none"] or ["none"]))

    targets, tol = {}, {}
    for k in objectives:
        t, tt = _sample_target(family.metric(k), tight, rng)
        targets[k] = t
        if family.metric(k).direction is Direction.TARGET:
            tol[k] = tt

    eff_min = max(d.min_objectives, len(required))
    span = max(1, d.max_objectives - eff_min)
    n_frac = (len(objectives) - eff_min) / span
    diff = 0.4 * (math.log2(tight) / 3.0) + 0.4 * max(0.0, n_frac) + 0.2 * (corner != "none")

    return GeneratedItem(
        item_id=f"{family.id}-{idx:04d}", family_id=family.id, tier=tier,
        tightness=tight, corner=corner, objectives=objectives, targets=targets,
        tolerance=tol, n_objectives=len(objectives), difficulty_score=round(diff, 3))


def generate_items(family: FamilySpec, n: int, rng: random.Random) -> list[GeneratedItem]:
    out = []
    for i in range(n):
        tier = rng.choices(TIERS, weights=TIER_WEIGHTS)[0]
        out.append(_gen_item(family, tier, i, rng))
    return out


def generate_bank(per_family: int = 60, seed: int = 0) -> list[GeneratedItem]:
    rng = random.Random(seed)
    bank = []
    for fam in FAMILIES.values():
        bank += generate_items(fam, per_family, rng)
    return bank


def mark_reliability_subset(bank: list[GeneratedItem], per_cell: int = 3,
                            seed: int = 1) -> list[GeneratedItem]:
    """Flag ~50 items spanning every (family × tier) cell for the paraphrase
    test-retest reliability study (Week 7)."""
    rng = random.Random(seed)
    chosen = []
    for fid in FAMILIES:
        for tier in TIERS:
            cell = [it for it in bank if it.family_id == fid and it.tier == tier]
            for it in rng.sample(cell, min(per_cell, len(cell))):
                it.reliability_subset = True
                chosen.append(it)
    return chosen


# --------------------------------------------------------------------------- #
# paraphrase generator — 2-3 surface forms, identical constraints
# --------------------------------------------------------------------------- #
_HZ = ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (1.0, "Hz"))


def _fmt(value, unit, style):
    if unit == "Hz":
        if style == 1:
            return f"{value:.3g} Hz"
        if style == 2:
            return f"{value/1e3:.4g} kHz"
        for s, name in _HZ:                     # style 0: SI prefix
            if value >= s:
                return f"{value/s:.3g} {name}"
        return f"{value:.3g} Hz"
    if unit == "W":
        if style == 1:
            return f"{value:.3g} W"
        if style == 2:
            return f"{value*1e6:.4g} µW"
        return (f"{value*1e3:.3g} mW" if value >= 1e-3 else f"{value*1e6:.3g} µW")
    if unit in ("V", "Vpp"):
        return (f"{value*1e3:.4g} m{unit}" if style == 1 else f"{value:.3g} {unit}")
    if unit == "dB":
        return f"{value:.0f} dB" if style == 2 else f"{value:.1f} dB"
    if unit == "deg":
        return f"{value:.0f} degrees" if style == 1 else f"{value:.0f}°"
    return f"{value:.2g}"


_GE = ("{lab} of at least {v}", "{lab} ≥ {v}", "{lab} no less than {v}")
_LE = ("{lab} of at most {v}", "{lab} ≤ {v}", "{lab} not exceeding {v}")
_TG = ("{lab} of {v} (±{tp}%)", "{lab} ≈ {v}, within {tp}%", "target {lab} {v} (±{tp}%)")
_WN = ("{lab} within {lo}–{hi}", "{lab} between {lo} and {hi}", "{lab} in [{lo}, {hi}]")
_INTRO = ("Design a {t}.", "Produce a SPICE netlist for a {t}.", "Implement a {t} circuit.")


def _clause(metric, target, tol, form):
    d, u, lab = metric.direction, metric.unit, metric.label
    if d is Direction.WINDOW:
        lo, hi = target
        return _WN[form].format(lab=lab, lo=_fmt(lo, u, form), hi=_fmt(hi, u, form))
    if d is Direction.GE:
        return _GE[form].format(lab=lab, v=_fmt(target, u, form))
    if d is Direction.LE:
        return _LE[form].format(lab=lab, v=_fmt(target, u, form))
    return _TG[form].format(lab=lab, v=_fmt(target, u, form), tp=round(tol * 100))


def render_prompt(item: GeneratedItem, family: FamilySpec, form: int) -> str:
    """One paraphrase surface form. form in {0,1,2}: different wording, unit
    phrasing, and constraint ordering — identical underlying constraints."""
    rng = random.Random(f"{item.item_id}:{form}")
    clauses = [_clause(family.metric(k), item.targets[k], item.tolerance.get(k, 0.0), form)
               for k in item.objectives]
    if form == 1:
        clauses = clauses[::-1]
    elif form == 2:
        rng.shuffle(clauses)
    reqs = "; ".join(clauses)
    intro = _INTRO[form].format(t=family.title.lower())
    nodes = ", ".join(f"{r}=`{n}`" for r, n in list(family.nodes.items())[:4])
    cond = ""
    vdd = family.test_conditions.get("vdd")
    if vdd:
        cond = f" Supply {_fmt(vdd, 'V', form)}."
    corner = (f" The design must hold across {CORNER_DESC.get(item.corner, item.corner)}."
              if item.corner != "none" else "")
    tail = ("Provide the SPICE netlist." if form != 2 else "Return only the netlist.")
    return f"{intro} Requirements: {reqs}.{cond}{corner} Nodes: {nodes}. {tail}"


# --------------------------------------------------------------------------- #
# self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    bank = generate_bank(per_family=60, seed=0)
    rel = mark_reliability_subset(bank)

    # invariants
    for it in bank:
        fam = FAMILIES[it.family_id]
        req = {m.key for m in fam.metrics if m.required and m.key in fam.difficulty.objective_pool}
        assert req <= set(it.objectives), f"{it.item_id}: missing required metric"
        assert set(it.targets) == set(it.objectives)
        assert all(k in it.objectives for k in it.tolerance)
        assert not (set(it.objectives) & UNIMPLEMENTED)
        assert len(it.objectives) <= fam.difficulty.max_objectives
        it.to_item_spec()                      # builds a scorable spec
        assert len(set(it.forms())) >= 2       # paraphrases are distinct

    from collections import Counter
    tiers = Counter(it.tier for it in bank)
    hard_tail = (tiers["hard"] + tiers["extreme"]) / len(bank)
    print(f"bank: {len(bank)} items across {len(FAMILIES)} families")
    print("tier mix:", dict(tiers), f"-> hard-tail fraction {hard_tail:.0%}")
    assert hard_tail > 0.55, "hard tail should be over-sampled"

    cells = {(it.family_id, it.tier) for it in rel}
    print(f"reliability subset: {len(rel)} items spanning "
          f"{len(cells)}/{len(FAMILIES)*len(TIERS)} (family×tier) cells")
    assert 40 <= len(rel) <= 60 and len(cells) == len(FAMILIES) * len(TIERS)

    ex = next(it for it in bank if it.tier == "extreme" and it.family_id == "two_stage_opamp")
    print(f"\nexample HARD item {ex.item_id}  tier={ex.tier} tightness={ex.tightness} "
          f"corner={ex.corner} difficulty={ex.difficulty_score}")
    print(f"  objectives={ex.objectives}")
    for i, p in enumerate(ex.forms()):
        print(f"  form {i}: {p}")

    print("\nOK: spec-template + paraphrase generators validated; "
          "hard tail over-sampled; reliability subset marked.")
