"""Locked spec schemas for the four circuit families (Week 2 scope lock).

This is the canonical, machine-readable definition consumed by:
  * the Week 3 spec-template generators (sample targets from `sample_range`,
    activate `objective_pool` subsets, apply `corner_options`),
  * the Week 2 `score(metrics, spec)` (each concrete item is a FamilySpec
    instance with targets filled in),
  * `extract_metrics` (each MetricSpec.analysis + .glossary says how to measure).

Difficulty axes (per the schedule) are encoded in `DifficultyAxes`:
  1. constraint-tightness multiplier `t` — scales how little slack a target allows.
       TARGET metric tolerance  = base_tol / t      (tighter window as t↑)
       GE floor / LE ceiling    = target chosen further into the hard end of
                                  sample_range as t↑ (exact map: Week 3 generator)
  2. number of simultaneous objectives — how many of `objective_pool` are active
       (sampled in [min_objectives, max_objectives]); more = harder.
  3. robustness/corner requirement — one of `corner_options`; the design must meet
       spec across that corner, not just nominal.

`docs/spec_glossary.md` holds the exact ngspice measurement recipe for every metric
(referenced by MetricSpec.glossary). `docs/families.md` is the human-readable view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Direction(str, Enum):
    GE = "ge"          # measured >= target  (floor: higher is better)
    LE = "le"          # measured <= target  (ceiling: lower is better)
    TARGET = "target"  # measured ~= target within tolerance (two-sided)
    WINDOW = "window"  # measured within [lo, hi] (target is the window itself)


# analyses an item may require; bound to harness routines in Week 2
ANALYSES = frozenset({"ac", "op", "dc", "tran", "ac_cmrr", "dc_icmr"})


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str
    direction: Direction
    sample_range: tuple[float, float]   # range the generator samples the target from
    analysis: str                       # one of ANALYSES
    glossary: str                       # section in docs/spec_glossary.md
    base_tol: float = 0.20              # nominal fractional tol (TARGET) at tightness=1
    required: bool = True               # always-on metric vs. optional objective
    notes: str = ""


@dataclass(frozen=True)
class DifficultyAxes:
    objective_pool: tuple[str, ...]                 # metric keys eligible as objectives
    min_objectives: int = 1
    max_objectives: int = 1
    corner_options: tuple[str, ...] = ("none",)
    tightness_levels: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class FamilySpec:
    id: str
    title: str
    description: str
    nodes: dict[str, str]               # role -> node name convention
    design_vars: tuple[str, ...]        # what the respondent sets
    test_conditions: dict               # supplies, source, load, analysis params
    metrics: tuple[MetricSpec, ...]
    difficulty: DifficultyAxes
    gates: tuple[str, ...] = ()          # hard validity gates (bias/region checks)

    def metric(self, key: str) -> MetricSpec:
        return next(m for m in self.metrics if m.key == key)


# --------------------------------------------------------------------------- #
# Family 1 — RC / RLC filters (passive)
# --------------------------------------------------------------------------- #
FILTERS = FamilySpec(
    id="filters",
    title="RC / RLC filters",
    description="Passive 1st/2nd-order low-pass, high-pass, or band-pass. Variant "
                "(LP/HP/BP, RC vs RLC) is a spec field; band-pass adds f0 + Q.",
    nodes={"input": "in", "output": "out", "ground": "0"},
    design_vars=("R", "C", "L (RLC only)"),
    test_conditions={
        "source": "V(in) AC 1",
        "ac_sweep": (1.0, 1e8, "dec", 50),   # f_start, f_stop, mode, ppd
        "load": "open (or RL/CL if specified)",
    },
    metrics=(
        MetricSpec("passband_gain_db", "Passband gain", "dB", Direction.WINDOW,
                   (-1.0, 0.0), "ac", "§1", required=True,
                   notes="passive insertion loss; window near 0 dB"),
        MetricSpec("cutoff_freq_hz", "−3 dB cutoff (LP/HP) / f0 (BP)", "Hz",
                   Direction.TARGET, (1e2, 1e5), "ac", "§2", base_tol=0.20,
                   required=True),
        MetricSpec("stopband_atten_db", "Attenuation at 10×fc", "dB", Direction.GE,
                   (15.0, 40.0), "ac", "§2", required=False,
                   notes="roll-off objective; ≥20 dB/dec ⇒ 1st order"),
        MetricSpec("q_factor", "Quality factor (band-pass)", "—", Direction.TARGET,
                   (0.5, 5.0), "ac", "§2", base_tol=0.25, required=False,
                   notes="BP only; Q = f0/BW from the two −3 dB edges"),
    ),
    difficulty=DifficultyAxes(
        objective_pool=("cutoff_freq_hz", "stopband_atten_db",
                        "passband_gain_db", "q_factor"),
        min_objectives=1, max_objectives=3,
        corner_options=("none", "comp_tol_5pct", "comp_tol_10pct"),
    ),
)

# --------------------------------------------------------------------------- #
# Family 2 — Common-source amplifier
# --------------------------------------------------------------------------- #
CS_AMP = FamilySpec(
    id="cs_amp",
    title="Common-source amplifier",
    description="Single MOSFET, resistive (or current-source) load, with bias. "
                "Small-signal gain/BW, large-signal swing, quiescent power.",
    nodes={"input": "in", "output": "out", "supply": "vdd", "ground": "0"},
    design_vars=("M1 W/L", "RD (load)", "bias (RS or Ibias)", "coupling/bypass caps"),
    test_conditions={
        "vdd": 1.8,
        "source": "V(in) DC=Vbias AC 1",
        "ac_sweep": (1.0, 1e9, "dec", 50),
        "dc_sweep": "Vin across input range (swing)",
        "load_cap_f": 1e-12,
    },
    metrics=(
        MetricSpec("gain_db", "Midband voltage gain", "dB", Direction.GE,
                   (15.0, 35.0), "ac", "§1"),
        MetricSpec("bw_3db_hz", "Upper −3 dB bandwidth", "Hz", Direction.TARGET,
                   (1e5, 1e8), "ac", "§2", base_tol=0.30, required=False),
        MetricSpec("out_swing_vpp", "Output swing", "Vpp", Direction.GE,
                   (0.4, 1.2), "dc", "§4", required=False),
        MetricSpec("quiescent_power_w", "Quiescent power", "W", Direction.LE,
                   (50e-6, 2e-3), "op", "§5", required=False),
    ),
    difficulty=DifficultyAxes(
        objective_pool=("gain_db", "bw_3db_hz", "out_swing_vpp", "quiescent_power_w"),
        min_objectives=1, max_objectives=4,
        corner_options=("none", "supply_pm10", "temp_0_85C", "vth_corner"),
    ),
    gates=("m1_saturation",),
)

# --------------------------------------------------------------------------- #
# Family 3 — Differential pair
# --------------------------------------------------------------------------- #
DIFF_PAIR = FamilySpec(
    id="diff_pair",
    title="Differential pair",
    description="Matched MOS pair, tail current source, resistive or mirror load. "
                "Single-ended output for a finite, meaningful CMRR.",
    nodes={"in_plus": "inp", "in_minus": "inm", "out": "out",
           "supply": "vdd", "ground": "0", "tail": "tail"},
    design_vars=("M1/M2 W/L", "tail current (Ibias / mirror)", "RD or mirror load"),
    test_conditions={
        "vdd": 1.8,
        "vcm_bias": 0.9,                       # nominal input common-mode
        "ac_sweep": (1.0, 5e7, "dec", 50),
        "cmrr_fref": 1e3,                      # sub-dominant-pole reference
        "icmr_sweep": (0.0, 1.8, 0.02),        # Vcm rail-to-rail, step
        "icmr_gain_frac": 0.5,
    },
    metrics=(
        MetricSpec("diff_gain_db", "Differential gain A_dm", "dB", Direction.GE,
                   (15.0, 40.0), "ac", "§1"),
        MetricSpec("cmrr_db", "CMRR", "dB", Direction.GE,
                   (40.0, 80.0), "ac_cmrr", "§6"),
        MetricSpec("icmr_v", "Input common-mode range", "V", Direction.GE,
                   (0.4, 1.2), "dc_icmr", "§7"),
        MetricSpec("quiescent_power_w", "Quiescent power", "W", Direction.LE,
                   (50e-6, 2e-3), "op", "§5", required=False),
        MetricSpec("bw_3db_hz", "Differential −3 dB bandwidth", "Hz",
                   Direction.TARGET, (1e5, 5e7), "ac", "§2", base_tol=0.30,
                   required=False),
    ),
    difficulty=DifficultyAxes(
        objective_pool=("diff_gain_db", "cmrr_db", "icmr_v",
                        "quiescent_power_w", "bw_3db_hz"),
        min_objectives=2, max_objectives=5,
        corner_options=("none", "tail_finite_ro", "mismatch_1pct", "supply_pm10"),
    ),
    gates=("m1_m2_saturation", "tail_saturation"),
)

# --------------------------------------------------------------------------- #
# Family 4 — Two-stage op-amp
# --------------------------------------------------------------------------- #
TWO_STAGE_OPAMP = FamilySpec(
    id="two_stage_opamp",
    title="Two-stage op-amp",
    description="Diff-pair input stage + common-source second stage + Miller "
                "compensation. Open-loop A0/GBW/phase-margin into a specified CL.",
    nodes={"in_plus": "inp", "in_minus": "inm", "out": "out",
           "supply": "vdd", "ground": "0"},
    design_vars=("input-pair sizes", "mirror/bias current", "2nd-stage device",
                 "Miller Cc (+ nulling Rz)", "load CL"),
    test_conditions={
        "vdd": 1.8,
        "config": "open-loop",
        "load_cap_f": 5e-12,                   # CL the PM/GBW are specified into
        "ac_sweep": (1.0, 1e9, "dec", 50),
        "dc_sweep": "Vin (swing)",
        "tran": "step input for slew (optional)",
    },
    metrics=(
        MetricSpec("dc_gain_db", "Open-loop DC gain A0", "dB", Direction.GE,
                   (60.0, 100.0), "ac", "§1"),
        MetricSpec("gbw_hz", "Gain-bandwidth product", "Hz", Direction.GE,
                   (1e6, 5e7), "ac", "§2/§3"),
        MetricSpec("phase_margin_deg", "Phase margin", "deg", Direction.GE,
                   (45.0, 65.0), "ac", "§3"),
        MetricSpec("quiescent_power_w", "Quiescent power", "W", Direction.LE,
                   (100e-6, 5e-3), "op", "§5", required=False),
        MetricSpec("out_swing_vpp", "Output swing", "Vpp", Direction.GE,
                   (0.6, 1.4), "dc", "§4", required=False),
        MetricSpec("slew_rate_vps", "Slew rate", "V/s", Direction.GE,
                   (1e5, 1e7), "tran", "§4 (tran; recipe TBD W3)",
                   required=False, notes="optional objective; tran recipe added W3"),
    ),
    difficulty=DifficultyAxes(
        objective_pool=("dc_gain_db", "gbw_hz", "phase_margin_deg",
                        "quiescent_power_w", "out_swing_vpp", "slew_rate_vps"),
        min_objectives=2, max_objectives=6,
        corner_options=("none", "cload_1_10pf", "supply_pm10", "temp_0_85C"),
    ),
    gates=("all_devices_saturation",),
)


FAMILIES: dict[str, FamilySpec] = {
    f.id: f for f in (FILTERS, CS_AMP, DIFF_PAIR, TWO_STAGE_OPAMP)
}


def _self_check() -> None:
    """Cheap structural validation — a 'verify everything' gate on the schema."""
    for fid, fam in FAMILIES.items():
        keys = {m.key for m in fam.metrics}
        assert len(keys) == len(fam.metrics), f"{fid}: duplicate metric keys"
        for m in fam.metrics:
            lo, hi = m.sample_range
            assert lo < hi, f"{fid}.{m.key}: bad sample_range {m.sample_range}"
            assert m.analysis in ANALYSES, f"{fid}.{m.key}: unknown analysis {m.analysis}"
            assert 0 < m.base_tol < 1, f"{fid}.{m.key}: base_tol out of range"
        d = fam.difficulty
        assert set(d.objective_pool) <= keys, f"{fid}: objective_pool not in metrics"
        assert 1 <= d.min_objectives <= d.max_objectives <= len(d.objective_pool), \
            f"{fid}: objective count bounds invalid"
        assert "none" in d.corner_options, f"{fid}: corner_options must include 'none'"
        # every required metric should also be reachable as an objective or always-on
        assert any(m.required for m in fam.metrics), f"{fid}: no required metric"
    print(f"OK: {len(FAMILIES)} families locked — "
          + ", ".join(f"{f.id}({len(f.metrics)} metrics)" for f in FAMILIES.values()))


if __name__ == "__main__":
    _self_check()
