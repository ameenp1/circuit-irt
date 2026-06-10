"""Unit-test the failure-mode classifier with hand-written correct + broken
netlists per family (incl. differential pair). Every broken netlist must get the
RIGHT label, not just a fail.

Run:  .venv/bin/python test_classifier.py
"""
from __future__ import annotations

import math

from circuit_irt.families import (Direction, FILTERS, CS_AMP, DIFF_PAIR, TWO_STAGE_OPAMP)
from circuit_irt.harness import (AnalysisPlan, ItemSpec, FailureMode, _to_lin,
                     simulate, extract_metrics, score, classify, fingerprint)

NMOS = ".model NM NMOS (LEVEL=1 VTO=0.45 KP=120u LAMBDA=0.02)"


def derive_spec(family, metrics, objectives) -> ItemSpec:
    """Targets set from the correct circuit's measured metrics so it passes with
    margin (GE target 0.7×, LE ceiling 1.5×, TARGET exact, WINDOW = family band)."""
    targets = {}
    for k in objectives:
        ms = family.metric(k); meas = metrics[k]
        if ms.direction is Direction.GE:
            lin = _to_lin(meas, ms.unit) * 0.7
            targets[k] = 20 * math.log10(lin) if ms.unit == "dB" else lin
        elif ms.direction is Direction.LE:
            targets[k] = meas * 1.5
        elif ms.direction is Direction.TARGET:
            targets[k] = meas
        else:  # WINDOW
            targets[k] = ms.sample_range
    return ItemSpec(family, targets, tuple(objectives))


def run(netlist, family, plan, item, reference_fp=None):
    raw = simulate(netlist, plan)
    metrics = extract_metrics(raw, family)
    s = score(metrics, item, syntax_ok=raw.syntax_ok, converged=raw.converged)
    label = classify(netlist, raw, s, reference_fp)
    return metrics, s, label


def check(name, expected, netlist, family, plan, item, ref=None):
    metrics, s, label = run(netlist, family, plan, item, ref)
    flag = "OK " if label is expected else "XX "
    fails = [k for k, ok in s["per_metric_pass"].items() if not ok]
    print(f"  {flag}{name:<26} -> {label.value:<28} "
          f"(fails={fails}, graded={s['graded']:.2f})")
    assert label is expected, f"{name}: got {label.value}, expected {expected.value}"


# =========================================================================== #
print("=== FILTERS (RC/RLC, passive) ===")
fp_plan = AnalysisPlan(("ac",), in_nodes=("in",), ac=(1, 1e7, "dec", 40), f_ref=10)
correct_rc = "R1 in out 1k\nC1 out 0 159.155n"          # fc = 1 kHz
m, _, _ = run(correct_rc, FILTERS, fp_plan,
              ItemSpec(FILTERS, {"cutoff_freq_hz": 1e3}, ("cutoff_freq_hz",)))
ref_fp_filt = fingerprint(correct_rc)
spec_filt = derive_spec(FILTERS, m, ("cutoff_freq_hz", "passband_gain_db",
                                     "stopband_atten_db"))
check("correct RC LP", FailureMode.PASS, correct_rc, FILTERS, fp_plan, spec_filt, ref_fp_filt)
check("R missing value", FailureMode.PARSE_FAILURE,
      "R1 in out\nC1 out 0 159n", FILTERS, fp_plan, spec_filt, ref_fp_filt)
check("resistive divider (no C)", FailureMode.WRONG_TOPOLOGY,
      "R1 in out 1k\nR2 out 0 1k", FILTERS, fp_plan, spec_filt, ref_fp_filt)
# single objective (cutoff) grossly off: C 10x too small -> fc 10x high
check("C 10x too small (gross)", FailureMode.MIS_SIZED,
      "R1 in out 1k\nC1 out 0 15.9n", FILTERS, fp_plan,
      ItemSpec(FILTERS, {"cutoff_freq_hz": 1e3}, ("cutoff_freq_hz",)), ref_fp_filt)

# =========================================================================== #
print("=== COMMON-SOURCE AMP (transistor) ===")
cs_plan = AnalysisPlan(("op", "ac"), in_nodes=("in",), input_bias=0.75,
                       supplies=(("VDD", "vdd", 1.8),), ac=(1, 1e9, "dec", 30),
                       f_ref=1e3, load_cap=1e-12)
correct_cs = f"{NMOS}\nM1 out in 0 0 NM W=100u L=1u\nRD vdd out 2k"
m_cs, _, _ = run(correct_cs, CS_AMP, cs_plan,
                 ItemSpec(CS_AMP, {"gain_db": 0}, ("gain_db",)))
print(f"     [correct cs: gain={m_cs['gain_db']:.1f}dB Pq={m_cs['quiescent_power_w']*1e3:.2f}mW]")
ref_fp_cs = fingerprint(correct_cs)
spec_cs = derive_spec(CS_AMP, m_cs, ("gain_db", "quiescent_power_w"))
check("correct CS amp", FailureMode.PASS, correct_cs, CS_AMP, cs_plan, spec_cs, ref_fp_cs)
check("v-source loop (singular)", FailureMode.NON_CONVERGENCE,
      correct_cs + "\nVl1 q 0 1\nVl2 q 0 2", CS_AMP, cs_plan, spec_cs, ref_fp_cs)
# single-objective gain: slightly low RD -> small gain shortfall (near-miss)
spec_cs_gain = derive_spec(CS_AMP, m_cs, ("gain_db",))
check("RD slightly low (near)", FailureMode.NEAR_MISS,
      f"{NMOS}\nM1 out in 0 0 NM W=100u L=1u\nRD vdd out 1.25k",
      CS_AMP, cs_plan, spec_cs_gain, ref_fp_cs)
check("RD tiny (gross)", FailureMode.MIS_SIZED,
      f"{NMOS}\nM1 out in 0 0 NM W=100u L=1u\nRD vdd out 120",
      CS_AMP, cs_plan, spec_cs_gain, ref_fp_cs)

# =========================================================================== #
print("=== DIFFERENTIAL PAIR (transistor) ===")
dp_plan = AnalysisPlan(("op", "ac", "ac_cmrr", "dc_icmr"),
                       in_nodes=("inp", "inm"), input_mode="differential",
                       input_bias=0.9, out_node="out",
                       supplies=(("VDD", "vdd", 1.8),), ac=(1, 5e7, "dec", 30),
                       f_ref=1e3, cmrr_fref=1e3, icmr=(0.2, 1.7, 0.05))
correct_dp = (f"{NMOS}\n"
              "M1 outp inp tail 0 NM W=200u L=1u\n"
              "M2 out  inm tail 0 NM W=200u L=1u\n"
              "RD1 vdd outp 2k\nRD2 vdd out 2k\n"
              "M3 tail nb 0 0 NM W=200u L=1u\nVnb nb 0 0.7")
m_dp, _, _ = run(correct_dp, DIFF_PAIR, dp_plan,
                 ItemSpec(DIFF_PAIR, {"diff_gain_db": 0}, ("diff_gain_db",)))
print(f"     [correct dp: Adm={m_dp['diff_gain_db']:.1f}dB "
      f"CMRR={m_dp['cmrr_db']:.1f}dB ICMR={m_dp['icmr_v']:.2f}V]")
ref_fp_dp = fingerprint(correct_dp)
spec_dp = derive_spec(DIFF_PAIR, m_dp, ("diff_gain_db", "cmrr_db", "icmr_v"))
check("correct diff pair", FailureMode.PASS, correct_dp, DIFF_PAIR, dp_plan, spec_dp, ref_fp_dp)
check("single transistor (no pair)", FailureMode.WRONG_TOPOLOGY,
      f"{NMOS}\nM1 out inp 0 0 NM W=200u L=1u\nRD2 vdd out 2k",
      DIFF_PAIR, dp_plan, spec_dp, ref_fp_dp)
# shrink the pair devices a lot: gain AND cmrr collapse -> >=2 fail
check("pair devices tiny (multi)", FailureMode.MULTI_OBJECTIVE_FAILURE,
      f"{NMOS}\n"
      "M1 outp inp tail 0 NM W=8u L=1u\nM2 out inm tail 0 NM W=8u L=1u\n"
      "RD1 vdd outp 2k\nRD2 vdd out 2k\n"
      "M3 tail nb 0 0 NM W=200u L=1u\nVnb nb 0 0.7",
      DIFF_PAIR, dp_plan, spec_dp, ref_fp_dp)

# =========================================================================== #
print("=== TWO-STAGE OP-AMP (behavioral) ===")
def opamp(a0=1000.0, p1=1e3, p2=3e6):
    c1 = 1 / (2 * math.pi * 1e3 * p1); c2 = 1 / (2 * math.pi * 1e3 * p2)
    return (f"E1 n1 0 inp inm {a0}\nR1 n1 n2 1k\nC1 n2 0 {c1:.6e}\n"
            f"E2 n3 0 n2 0 1\nR2 n3 out 1k\nC2 out 0 {c2:.6e}")
oa_plan = AnalysisPlan(("ac",), in_nodes=("inp", "inm"), input_mode="differential",
                       ac=(1, 1e8, "dec", 50), f_ref=10)
correct_oa = opamp()
m_oa, _, _ = run(correct_oa, TWO_STAGE_OPAMP, oa_plan,
                 ItemSpec(TWO_STAGE_OPAMP, {"dc_gain_db": 0}, ("dc_gain_db",)))
print(f"     [correct op-amp: A0={m_oa['dc_gain_db']:.1f}dB "
      f"GBW={m_oa['gbw_hz']:.3g}Hz PM={m_oa['phase_margin_deg']:.1f}deg]")
ref_fp_oa = fingerprint(correct_oa)
spec_oa = derive_spec(TWO_STAGE_OPAMP, m_oa, ("dc_gain_db", "gbw_hz", "phase_margin_deg"))
check("correct op-amp", FailureMode.PASS, correct_oa, TWO_STAGE_OPAMP, oa_plan, spec_oa, ref_fp_oa)
# single-objective PM: pull 2nd pole in a touch -> PM just below target (near-miss)
spec_oa_pm = derive_spec(TWO_STAGE_OPAMP, m_oa, ("phase_margin_deg",))
check("2nd pole closer (near PM)", FailureMode.NEAR_MISS,
      opamp(p2=7e5), TWO_STAGE_OPAMP, oa_plan, spec_oa_pm, ref_fp_oa)
check("2nd pole at p1 (gross PM)", FailureMode.MIS_SIZED,
      opamp(p2=2e3), TWO_STAGE_OPAMP, oa_plan, spec_oa_pm, ref_fp_oa)

print("\nOK: every correct netlist passes and every broken netlist gets its right label.")
