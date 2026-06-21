"""Week 4 spot-audit of the failure-mode classifier (+ parser hardening).

~30 hand-chosen (completion, item) cases spanning every category and the messy
formats real 1-3B models emit. Prints a review table for by-hand inspection,
asserts the expected label on each, and logs the category distribution to
data/failure_audit.json.

Merge decision from this audit (applied in circuit_irt.harness.classify):
  * PASS is now checked BEFORE WRONG_TOPOLOGY — a functionally-correct design is
    a pass even if its topology differs from our single reference. WRONG_TOPOLOGY
    is reserved for designs that FAIL and don't match the reference shape.
  * parse_failure keeps one label but logs a sub-reason: `no_netlist` (respondent
    emitted nothing usable) vs `invalid_spice` (ngspice rejected the netlist).
"""
from __future__ import annotations

import json
import math

from circuit_irt.paths import DATA
from circuit_irt.reference import NMOS
from circuit_irt.respondent import evaluate_completion, FailureLog

items = json.load(open(DATA / "candidate_bank.json"))["items"]
FS = next(i for i in items if i["family_id"] == "filters"
          and "stopband_atten_db" in i["objectives"])
FE = next(i for i in items if i["family_id"] == "filters"
          and set(i["objectives"]) == {"cutoff_freq_hz", "passband_gain_db"})
CS = next(i for i in items if i["family_id"] == "cs_amp"
          and {"gain_db", "bw_3db_hz"} <= set(i["objectives"]))
DP = next(i for i in items if i["family_id"] == "diff_pair")
BOGUS = {**FS, "objectives": ["bogus_metric"]}    # triggers the never-crash guard

jwrap = lambda n: json.dumps({"netlist": n})


def rlc(fc):                                   # same 2nd-order Butterworth as references
    w = 2 * math.pi * fc; C = 100e-9; L = 1 / (w * w * C); R = math.sqrt(L / C) / 0.707
    return f"R1 in n1 {R:.6g}\nL1 n1 out {L:.6g}\nC1 out 0 {C:.6g}"


def rc(fc):                                    # 1st-order RC (different topology)
    R = 1e3; C = 1 / (2 * math.pi * R * fc)
    return f"R1 in out {R:.6g}\nC1 out 0 {C:.6g}"


cs_lowgain = f"{NMOS}\nM1 out in 0 0 NM W=100u L=1u\nRD vdd out 200"   # same topo, tanked
fc_fs = FS["targets"]["cutoff_freq_hz"]
fc_fe = FE["targets"]["cutoff_freq_hz"]

# (name, completion, item, expected_label, expected_reason|None)  — None reason = unchecked
CASES = [
    # ---- clean PASS + JSON-format robustness (all the correct RLC, recovered) ----
    ("clean json",            jwrap(rlc(fc_fs)), FS, "pass", None),
    ("trailing-comma json",   '{"netlist": "' + rlc(fc_fs).replace("\n", "\\n") + '",}', FS, "pass", None),
    ("single-quoted json",    "{'netlist': '" + rlc(fc_fs).replace("\n", "\\n") + "'}", FS, "pass", None),
    ("real newlines in json", '{"netlist": "' + rlc(fc_fs) + '"}', FS, "pass", None),
    ("```json fence",         f"```json\n{jwrap(rlc(fc_fs))}\n```", FS, "pass", None),
    ("```spice fence",        f"```spice\n{rlc(fc_fs)}\n```", FS, "pass", None),
    ("bare netlist",          rlc(fc_fs), FS, "pass", None),
    ("prose + json",          f"Here's the design.\n{jwrap(rlc(fc_fs))}\nHope it helps!", FS, "pass", None),
    ("reasoning <think> wrap", f"<think>RLC low-pass needed</think>{jwrap(rlc(fc_fs))}", FS, "pass", None),
    ("unterminated ```fence",  f"```\n{rlc(fc_fs)}", FS, "pass", None),

    # ---- WRONG_TOPOLOGY: only when it also FAILS (audit reorder) ----
    ("alt topology PASSES (RC on easy item)", jwrap(rc(fc_fe)), FE, "pass", None),
    ("wrong topology FAILS (divider)", jwrap("R1 in out 1k\nR2 out 0 1k"), FS, "wrong_topology", None),

    # ---- graded failure modes (same topology as reference) ----
    ("near-miss (cutoff 1.3x)", jwrap(rlc(fc_fs * 1.3)), FS, "single_constraint_near_miss", None),
    ("mis-sized (cutoff 2.2x)", jwrap(rlc(fc_fs * 2.2)), FS, "topology_correct_mis_sized", None),
    ("multi-objective (CS RD tanked)", jwrap(cs_lowgain), CS, "multi_objective_failure", None),

    # ---- non-convergence (fail not crash) ----
    ("v-source loop (singular)", jwrap(rlc(fc_fs) + "\nVl1 q 0 1\nVl2 q 0 2"), FS, "non_convergence", None),

    # ---- parse_failure: invalid SPICE (parsed text, ngspice rejects) ----
    ("real 0.5B garbage", jwrap("M1 in M1 out 0 0 NM W=80u L=1u\nR1 1k R2 1k"), FS, "parse_failure", "invalid_spice"),
    ("malformed element", jwrap("R1 in\nC1 out"), FS, "parse_failure", "no_netlist"),

    # ---- parse_failure: no netlist at all ----
    ("refusal",      "I'm sorry, I can't help with that.", FS, "parse_failure", "no_netlist"),
    ("empty",        "", FS, "parse_failure", "no_netlist"),
    ("pure prose",   "First, consider the dominant pole and choose components accordingly.", FS, "parse_failure", "no_netlist"),

    # ---- truncated mid-netlist (token cutoff): must not crash ----
    ("truncated json string", '{"netlist": "' + rlc(fc_fs).replace("\n", "\\n")[:-8], FS, None, None),

    # ---- model adds supply/stimulus/control (stripped) ----
    ("model adds supply+ctrl", jwrap("VDD vdd 0 1.8\n" + rlc(fc_fs) + "\n.ac dec 10 1 1e6\n.end"), FS, "pass", None),

    # ---- more format robustness ----
    ("extra json fields", json.dumps({"reasoning": "use Butterworth", "netlist": rlc(fc_fs)}), FS, "pass", None),
    ("multiple code blocks", f"```python\nprint('hi')\n```\n```json\n{jwrap(rlc(fc_fs))}\n```", FS, "pass", None),
    ("markdown backticks/bold", f"**Netlist:**\n`{rlc(fc_fs).splitlines()[0]}`\n" + "\n".join(rlc(fc_fs).splitlines()[1:]), FS, "pass", None),

    # ---- differential pair (heavy family: cmrr + icmr) ----
    ("diff-pair reference PASSES", jwrap(DP["reference_netlist"]), DP, "pass", None),
    ("diff-pair wrong topo (single FET)", jwrap(f"{NMOS}\nM1 out inp 0 0 NM W=200u L=1u\nRD2 vdd out 2k"), DP, "wrong_topology", None),

    # ---- never-crash guarantee: malformed item -> harness_error, not a crash ----
    ("malformed item (bad objective)", jwrap(rlc(fc_fs)), BOGUS, "harness_error", None),
]


def run():
    log = FailureLog()
    print(f"items: FS={FS['item_id']}({FS['objectives']})  "
          f"FE={FE['item_id']}  CS={CS['item_id']}\n")
    hdr = f"{'#':>2}  {'case':<34} {'parsed':<6} {'label':<28} {'reason':<14} graded"
    print(hdr); print("-" * len(hdr))
    fails = 0
    for i, (name, comp, item, exp_label, exp_reason) in enumerate(CASES, 1):
        r = log.record(evaluate_completion(comp, item))      # never raises
        ok = ((exp_label is None or r["label"] == exp_label)
              and (exp_reason is None or r["reason"] == exp_reason))
        fails += not ok
        print(f"{i:>2}  {'OK ' if ok else 'XX '}{name:<31} {r['parsed']!s:<6} "
              f"{r['label']:<28} {str(r['reason'] or ''):<14} {r['graded']:.2f}")
        assert ok, (f"{name}: got label={r['label']} reason={r['reason']}, "
                    f"want {exp_label}/{exp_reason}")

    print("\ncategory distribution:", log.summary()["categories"])
    print("sub-reasons:", log.summary()["reasons"])
    log.to_json(DATA / "failure_audit.json")
    print(f"\nlogged {log.summary()['n']} audited labels -> data/failure_audit.json")
    assert fails == 0
    print("OK: 30-case audit — every label defensible; PASS-before-topology merge applied.")


if __name__ == "__main__":
    run()
