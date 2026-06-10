"""Memorization probe set (Week 3).

Canonical textbook circuits, each in a VERBATIM form (canonical component values
-> round, recognizable target spec) and a COMPONENT-VALUE-PERTURBED form (same
topology, shifted values -> shifted spec). Week 7 scores the verbatim-vs-perturbed
gap: a model that *recalled* the textbook netlist passes verbatim but fails
perturbed; a model that genuinely *designs* passes both. The gap is the
memorization signal.

Validity property each probe must satisfy (asserted in __main__):
  * canonical netlist PASSES its verbatim spec   (it is the textbook answer)
  * canonical netlist FAILS its perturbed spec   (recall no longer suffices)
  * perturbed netlist PASSES its perturbed spec  (the shifted spec is achievable)
  * fingerprint(canonical) == fingerprint(perturbed)  (only values changed)

Families covered: RC low-pass filters, common-source amp, differential pair —
the circuits with genuinely recallable canonical netlists. (Op-amp probes are
deferred until a verified transistor Miller reference exists; a behavioral op-amp
is not something a model would "recall", so it is not a valid memorization probe.)
"""
from __future__ import annotations

from dataclasses import dataclass

from circuit_irt.families import FAMILIES
from circuit_irt.generators import spec_from_metrics, _clause
from circuit_irt.harness import (AnalysisPlan, simulate, extract_metrics, score, fingerprint)

NMOS = ".model NM NMOS (LEVEL=1 VTO=0.45 KP=120u LAMBDA=0.02)"

PLANS = {
    "filters": AnalysisPlan(("ac",), in_nodes=("in",), ac=(1, 1e7, "dec", 40), f_ref=10),
    "cs_amp": AnalysisPlan(("op", "ac"), in_nodes=("in",), input_bias=0.78,
                           supplies=(("VDD", "vdd", 1.8),), ac=(1, 1e9, "dec", 30),
                           f_ref=1e3, load_cap=1e-12),
    "diff_pair": AnalysisPlan(("ac",), in_nodes=("inp", "inm"), input_mode="differential",
                              input_bias=0.9, out_node="out", supplies=(("VDD", "vdd", 1.8),),
                              ac=(1, 5e8, "dec", 30), f_ref=1e3, load_cap=5e-12),
}


@dataclass
class Probe:
    pid: str
    name: str
    family_id: str
    canonical: str
    perturbed: str
    objectives: tuple
    disc: str            # discriminating TARGET metric the perturbation shifts


def _cs(w, rd, l="1u"):
    return f"{NMOS}\nM1 out in 0 0 NM W={w} L={l}\nRD vdd out {rd}"


def _dp(w, rd, vnb="0.7"):
    return (f"{NMOS}\n"
            f"M1 outp inp tail 0 NM W={w} L=1u\nM2 out inm tail 0 NM W={w} L=1u\n"
            f"RD1 vdd outp {rd}\nRD2 vdd out {rd}\n"
            f"M3 tail nb 0 0 NM W=200u L=1u\nVnb nb 0 {vnb}")


PROBES = [
    # --- RC low-pass filters at round textbook corners (perturb C or R) -------
    Probe("mem-f01", "RC LP, fc=1 kHz (1k/159n)", "filters",
          "R1 in out 1k\nC1 out 0 159.155n", "R1 in out 1k\nC1 out 0 318.31n",
          ("cutoff_freq_hz", "passband_gain_db"), "cutoff_freq_hz"),
    Probe("mem-f02", "RC LP, fc=1 kHz (10k/15.9n)", "filters",
          "R1 in out 10k\nC1 out 0 15.9155n", "R1 in out 10k\nC1 out 0 7.95775n",
          ("cutoff_freq_hz", "passband_gain_db"), "cutoff_freq_hz"),
    Probe("mem-f03", "RC LP, fc=100 Hz (1k/1.59u)", "filters",
          "R1 in out 1k\nC1 out 0 1.59155u", "R1 in out 2k\nC1 out 0 1.59155u",
          ("cutoff_freq_hz", "passband_gain_db"), "cutoff_freq_hz"),
    Probe("mem-f04", "RC LP, fc=10 kHz (1k/15.9n)", "filters",
          "R1 in out 1k\nC1 out 0 15.9155n", "R1 in out 1k\nC1 out 0 53.05n",
          ("cutoff_freq_hz", "passband_gain_db"), "cutoff_freq_hz"),
    Probe("mem-f05", "RC LP, fc=1 kHz (16k/9.95n)", "filters",
          "R1 in out 16k\nC1 out 0 9.947n", "R1 in out 8k\nC1 out 0 9.947n",
          ("cutoff_freq_hz", "passband_gain_db"), "cutoff_freq_hz"),
    Probe("mem-f06", "RC LP, fc=500 Hz (3.18k/100n)", "filters",
          "R1 in out 3.183k\nC1 out 0 100n", "R1 in out 3.183k\nC1 out 0 40n",
          ("cutoff_freq_hz", "passband_gain_db"), "cutoff_freq_hz"),
    # --- common-source amplifiers (perturb RD -> shifts gain + bandwidth) -----
    Probe("mem-c01", "CS amp, W=100u RD=2k", "cs_amp",
          _cs("100u", "2k"), _cs("100u", "1.1k"),
          ("gain_db", "bw_3db_hz"), "bw_3db_hz"),
    Probe("mem-c02", "CS amp, W=200u RD=1.5k", "cs_amp",
          _cs("200u", "1.5k"), _cs("200u", "0.85k"),
          ("gain_db", "bw_3db_hz"), "bw_3db_hz"),
    Probe("mem-c03", "CS amp, W=50u RD=3k", "cs_amp",
          _cs("50u", "3k"), _cs("50u", "1.7k"),
          ("gain_db", "bw_3db_hz"), "bw_3db_hz"),
    # --- differential pairs (perturb both RD -> shifts gain + bandwidth) ------
    Probe("mem-d01", "Diff pair, W=200u RD=2k", "diff_pair",
          _dp("200u", "2k"), _dp("200u", "1.1k"),
          ("diff_gain_db", "bw_3db_hz"), "bw_3db_hz"),
    Probe("mem-d02", "Diff pair, W=150u RD=2.5k", "diff_pair",
          _dp("150u", "2.5k"), _dp("150u", "1.4k"),
          ("diff_gain_db", "bw_3db_hz"), "bw_3db_hz"),
    Probe("mem-d03", "Diff pair, W=200u RD=2k, Vb=0.75", "diff_pair",
          _dp("200u", "2k", "0.75"), _dp("200u", "1.2k", "0.75"),
          ("diff_gain_db", "bw_3db_hz"), "bw_3db_hz"),
]


def _measure(netlist, family_id):
    raw = simulate(netlist, PLANS[family_id])
    return extract_metrics(raw, FAMILIES[family_id]), raw


def probe_prompt(family, spec) -> str:
    clauses = [_clause(family.metric(k), spec.targets[k], spec.tolerance.get(k, 0.0), 0)
               for k in spec.objectives]
    return f"Design a {family.title.lower()} meeting: " + "; ".join(clauses) + "."


def build_probes() -> list[dict]:
    """Realize every probe: derive verbatim/perturbed specs from the canonical and
    perturbed references, and verify the memorization contrast."""
    fam = lambda f: FAMILIES[f]
    out = []
    for p in PROBES:
        family = fam(p.family_id)
        m_can, raw_can = _measure(p.canonical, p.family_id)
        m_per, raw_per = _measure(p.perturbed, p.family_id)
        # verbatim spec from canonical metrics; perturbed spec from perturbed metrics
        spec_v = spec_from_metrics(family, m_can, p.objectives)
        spec_p = spec_from_metrics(family, m_per, p.objectives)
        s_can_v = score(m_can, spec_v, syntax_ok=raw_can.syntax_ok, converged=raw_can.converged)
        s_per_p = score(m_per, spec_p, syntax_ok=raw_per.syntax_ok, converged=raw_per.converged)
        s_can_p = score(m_can, spec_p)          # canonical answer vs perturbed spec
        out.append(dict(
            probe=p, family=family,
            m_can=m_can, m_per=m_per,
            verbatim_spec=spec_v, perturbed_spec=spec_p,
            verbatim_prompt=probe_prompt(family, spec_v),
            perturbed_prompt=probe_prompt(family, spec_p),
            canonical_passes_verbatim=s_can_v["all_pass"],
            perturbed_passes_perturbed=s_per_p["all_pass"],
            canonical_fails_perturbed=not s_can_p["all_pass"],
            disc_failed=not s_can_p["per_metric_pass"].get(p.disc, True),
        ))
    return out


if __name__ == "__main__":
    recs = build_probes()
    print(f"{len(recs)} memorization probes "
          f"({sum(1 for r in recs if r['probe'].family_id=='filters')} filters, "
          f"{sum(1 for r in recs if r['probe'].family_id=='cs_amp')} CS, "
          f"{sum(1 for r in recs if r['probe'].family_id=='diff_pair')} diff pair)\n")
    hdr = f"{'probe':<9} {'disc metric':<14} {'verbatim':>11} {'perturbed':>11}  contrast"
    print(hdr); print("-" * len(hdr))
    for r in recs:
        p = r["probe"]; d = p.disc
        vt = "OK" if r["canonical_passes_verbatim"] else "XX"
        pt = "OK" if r["perturbed_passes_perturbed"] else "XX"
        cf = "OK" if (r["canonical_fails_perturbed"] and r["disc_failed"]) else "XX"
        print(f"{p.pid:<9} {d:<14} {r['m_can'][d]:>11.4g} {r['m_per'][d]:>11.4g}  "
              f"can→verb {vt} | pert→pert {pt} | can→pert fails {cf}")
        # validity assertions
        assert fingerprint(p.canonical) == fingerprint(p.perturbed), f"{p.pid}: topology changed"
        assert r["canonical_passes_verbatim"], f"{p.pid}: canonical fails its verbatim spec"
        assert r["perturbed_passes_perturbed"], f"{p.pid}: perturbed fails its perturbed spec"
        assert r["canonical_fails_perturbed"] and r["disc_failed"], \
            f"{p.pid}: NO memorization contrast (canonical also satisfies perturbed spec)"

    ex = recs[0]
    print(f"\nexample {ex['probe'].pid} ({ex['probe'].name}):")
    print(f"  verbatim : {ex['verbatim_prompt']}")
    print(f"  perturbed: {ex['perturbed_prompt']}")
    print("\nOK: all probes verified — canonical passes verbatim, fails perturbed; "
          "perturbation preserves topology.")
