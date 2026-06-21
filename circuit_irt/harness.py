"""Grading harness v0: simulate -> extract_metrics -> score.

  * simulate(netlist, plan)      -> RawOutput   (wraps `ngspice -b`; timeout +
                                                  explicit convergence capture)
  * extract_metrics(raw, family) -> {metric: value}   (per docs/spec_glossary.md)
  * score(metrics, item)         -> {per_metric_pass, graded, ...}

Measurement recipes live in docs/spec_glossary.md; metric directions / units /
valid ranges live in families.py. The harness OWNS the test fixture (supplies,
standardized stimulus, load) so every respondent design is measured identically —
this is a measurement harness, not a "does it run" checker.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from circuit_irt.families import Direction, FamilySpec

DB_TO_LN = math.log(10) / 20.0          # dB shortfall -> natural-log ratio (voltage)
_3DB = 3.0102999566                     # 20*log10(sqrt(2))


# --------------------------------------------------------------------------- #
# status + containers
# --------------------------------------------------------------------------- #
class SimStatus(str, Enum):
    OK = "ok"
    NONCONVERGENCE = "nonconvergence"
    SINGULAR_MATRIX = "singular_matrix"
    TIMESTEP = "timestep_too_small"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    NO_OUTPUT = "no_output"
    NGSPICE_ERROR = "ngspice_error"


_CONVERGED_BAD = {SimStatus.NONCONVERGENCE, SimStatus.SINGULAR_MATRIX,
                  SimStatus.TIMESTEP, SimStatus.TIMEOUT, SimStatus.NO_OUTPUT}
_SYNTAX_BAD = {SimStatus.SYNTAX_ERROR, SimStatus.NGSPICE_ERROR}


@dataclass
class AnalysisPlan:
    """Test fixture + analysis directives. Built from a family's test_conditions."""
    analyses: tuple[str, ...]
    in_nodes: tuple[str, ...] = ("in",)            # ("in",) or ("inp","inm")
    out_expr: str = "v(out)"
    out_node: str = "out"                          # node for load cap
    input_mode: str = "single"                     # "single" | "differential"
    input_bias: float = 0.0                        # DC bias / common-mode (Vcm)
    supplies: tuple[tuple[str, str, float], ...] = ()   # (elem, node, value)
    ac: tuple[float, float, str, int] = (1.0, 1e9, "dec", 50)
    f_ref: float = 10.0
    dc_sweep: tuple[float, float, float] = (-1.0, 1.0, 0.01)
    cmrr_fref: float = 1e3
    icmr: tuple[float, float, float] = (0.0, 1.8, 0.05)
    icmr_gain_frac: float = 0.5
    load_cap: float | None = None
    timeout: float = 30.0


@dataclass
class AnalysisResult:
    name: str
    status: SimStatus
    data: dict = field(default_factory=dict)
    stdout: str = ""


@dataclass
class RawOutput:
    plan: AnalysisPlan
    results: dict[str, AnalysisResult] = field(default_factory=dict)
    syntax_ok: bool = True
    converged: bool = True
    status: SimStatus = SimStatus.OK
    stdout: str = ""


# --------------------------------------------------------------------------- #
# ngspice invocation + convergence classification
# --------------------------------------------------------------------------- #
def _invoke(deck: str, timeout: float, wd: Path) -> tuple[int, str]:
    exe = shutil.which("ngspice")
    if exe is None:
        raise RuntimeError("ngspice not found on PATH")
    cir = wd / "deck.cir"
    cir.write_text(deck)
    try:
        p = subprocess.run([exe, "-b", cir.name], cwd=wd,
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return -9, "__TIMEOUT__"


def _classify(rc: int, blob: str, produced: bool) -> SimStatus:
    if blob == "__TIMEOUT__":
        return SimStatus.TIMEOUT
    low = blob.lower()
    # Parse/syntax FIRST: a malformed or dropped element is a parse failure even
    # though it then manifests as a singular matrix. ngspice flags these as
    # *warnings* ("...is not a valid resistor instance line, ignored!").
    if any(s in low for s in ("error on line", "is not a valid", "instance line",
                              "ignored!", "unrecognized", "unknown subckt",
                              "no such param", "syntax error", "could not find",
                              "unknown parameter", "bad node", "not enough")):
        return SimStatus.SYNTAX_ERROR
    if "singular matrix" in low:
        return SimStatus.SINGULAR_MATRIX
    if "timestep too small" in low:
        return SimStatus.TIMESTEP
    if any(s in low for s in ("no convergence", "iteration limit",
                              "convergence problems", "gmin stepping failed",
                              "source stepping failed", "supplies reduced")):
        return SimStatus.NONCONVERGENCE
    # If the expected output was produced and no error pattern fired, it's a
    # success — ngspice's exit code is unreliable (Ubuntu builds return rc=1 on a
    # clean batch run with a benign "no .plot/.print" note), so rc only matters
    # when nothing was produced.
    if produced:
        return SimStatus.OK
    if rc != 0:
        return SimStatus.NGSPICE_ERROR
    return SimStatus.NO_OUTPUT


def _load(path: Path, ncols: int) -> np.ndarray | None:
    """Tolerant wrdata reader: returns (n, ncols) or None if absent/malformed."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        a = np.loadtxt(path)
    except Exception:
        return None
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a if a.shape[1] == ncols else None


# --------------------------------------------------------------------------- #
# fixture / deck construction
# --------------------------------------------------------------------------- #
def _supplies(plan: AnalysisPlan) -> list[str]:
    return [f"{e} {n} 0 DC {v}" for (e, n, v) in plan.supplies]


def _inputs(plan: AnalysisPlan, *, ac: bool, exc=None, vcm=None) -> list[str]:
    L = []
    if plan.input_mode == "single":
        bias = plan.input_bias if vcm is None else vcm
        acs = " AC 1" if ac else ""
        L.append(f"Vstim {plan.in_nodes[0]} 0 DC {bias}{acs}")
    else:                                            # differential, via cm node
        vc = plan.input_bias if vcm is None else vcm
        L.append(f"Vcmbias cm 0 DC {vc}")
        ap, php, am, phm = exc if (ac and exc) else (0, 0, 0, 0)
        acp = f" AC {ap} {php}" if ac else ""
        acm = f" AC {am} {phm}" if ac else ""
        L.append(f"Vsp {plan.in_nodes[0]} cm DC 0{acp}")
        L.append(f"Vsm {plan.in_nodes[1]} cm DC 0{acm}")
    if plan.load_cap:
        L.append(f"Cload {plan.out_node} 0 {plan.load_cap}")
    return L


def _deck(circuit: str, fixture: list[str], control: list[str]) -> str:
    # NB: SPICE always treats the first line as the title/comment — must be a
    # comment or the first real element line is silently dropped.
    body = ["* harness deck", circuit.rstrip(), *fixture,
            ".control", *control, ".endc", ".end"]
    return "\n".join(body) + "\n"


# excitations for differential AC
_DM = (0.5, 0, 0.5, 180)     # v_id = 1, v_ic = 0
_CM = (1.0, 0, 1.0, 0)       # v_ic = 1, v_id = 0


# --------------------------------------------------------------------------- #
# simulate
# --------------------------------------------------------------------------- #
def simulate(netlist: str, plan: AnalysisPlan) -> RawOutput:
    """Run each requested analysis through `ngspice -b`; capture convergence."""
    out = RawOutput(plan=plan)
    wd = Path(tempfile.mkdtemp(prefix="harness_"))
    stdout_parts = []

    def run(name: str, fixture: list[str], control: list[str],
            outfiles: list[str]) -> AnalysisResult:
        deck = _deck(netlist, fixture, control)
        rc, blob = _invoke(deck, plan.timeout, wd)
        produced = any((wd / f).exists() and (wd / f).stat().st_size for f in outfiles)
        st = _classify(rc, blob, produced)
        stdout_parts.append(f"--- {name} [{st.value}] ---\n{blob}")
        return AnalysisResult(name=name, status=st, stdout=blob)

    for a in plan.analyses:
        f1, f2, mode, ppd = plan.ac
        if a == "ac":
            fx = _supplies(plan) + _inputs(plan, ac=True,
                                           exc=_DM if plan.input_mode == "differential" else None)
            r = run("ac", fx, [f"ac {mode} {ppd} {f1} {f2}",
                               f"let hd = db({plan.out_expr})",
                               f"let hp = 180*cph({plan.out_expr})/pi",
                               "wrdata ac.txt hd hp"], ["ac.txt"])
            d = _load(wd / "ac.txt", 4)
            if d is not None:
                r.data = {"freq": d[:, 0], "db": d[:, 1], "phase_deg": d[:, 3]}
            out.results["ac"] = r

        elif a == "op":
            branches = [f"{e.lower()}#branch" for (e, _, _) in plan.supplies]
            ctrl = ["op", f"wrdata op.txt {' '.join(branches)}"] if branches else ["op"]
            fx = _supplies(plan) + _inputs(plan, ac=False)
            r = run("op", fx, ctrl, ["op.txt"])
            d = _load(wd / "op.txt", 2 * len(branches)) if branches else None
            if d is not None:
                # wrdata repeats scale per vector -> value cols are odd indices
                vals = {plan.supplies[i][0]: float(d[0, 2 * i + 1])
                        for i in range(len(branches))}
                r.data = {"branch": vals}
            out.results["op"] = r

        elif a == "dc":
            lo, hi, step = plan.dc_sweep
            fx = _supplies(plan) + _inputs(plan, ac=False)
            r = run("dc", fx, [f"dc Vstim {lo} {hi} {step}", "wrdata dc.txt v(out)"],
                    ["dc.txt"])
            d = _load(wd / "dc.txt", 2)
            if d is not None:
                r.data = {"vin": d[:, 0], "vout": d[:, 1]}
            out.results["dc"] = r

        elif a == "ac_cmrr":
            data = {}
            st_worst = SimStatus.OK
            for tag, exc in (("dm", _DM), ("cm", _CM)):
                fx = _supplies(plan) + _inputs(plan, ac=True, exc=exc)
                rr = run(f"ac_cmrr_{tag}", fx,
                         [f"ac {mode} {ppd} {f1} {f2}",
                          f"let g = db({plan.out_expr})", f"wrdata {tag}.txt g"],
                         [f"{tag}.txt"])
                dd = _load(wd / f"{tag}.txt", 2)
                if dd is not None:
                    data[tag] = {"freq": dd[:, 0], "db": dd[:, 1]}
                if rr.status != SimStatus.OK:
                    st_worst = rr.status
            out.results["ac_cmrr"] = AnalysisResult("ac_cmrr", st_worst, data)

        elif a == "dc_icmr":
            vlo, vhi, dv = plan.icmr
            vcms, adm = [], []
            st_worst = SimStatus.OK
            for vcm in np.arange(vlo, vhi + 1e-9, dv):
                fx = _supplies(plan) + _inputs(plan, ac=True, exc=_DM, vcm=float(vcm))
                rr = run("dc_icmr_pt", fx,
                         [f"ac lin 1 {plan.cmrr_fref} {plan.cmrr_fref}",
                          f"let g = mag({plan.out_expr})", "wrdata icmr.txt g"],
                         ["icmr.txt"])
                dd = _load(wd / "icmr.txt", 2)
                if dd is not None:
                    vcms.append(float(vcm)); adm.append(float(dd[0, 1]))
                elif rr.status != SimStatus.OK:
                    st_worst = rr.status
            out.results["dc_icmr"] = AnalysisResult(
                "dc_icmr", st_worst,
                {"vcm": np.array(vcms), "adm": np.array(adm)})

    # roll up overall status
    sts = [r.status for r in out.results.values()]
    out.syntax_ok = not any(s in _SYNTAX_BAD for s in sts)
    out.converged = not any(s in (_CONVERGED_BAD | _SYNTAX_BAD) for s in sts)
    out.status = next((s for s in sts if s != SimStatus.OK), SimStatus.OK)
    out.stdout = "\n".join(stdout_parts)
    return out


# --------------------------------------------------------------------------- #
# extract_metrics
# --------------------------------------------------------------------------- #
def _interp_db(freq, db, f):
    return float(np.interp(math.log10(f), np.log10(freq), db))


def _logf_cross(freq, y, target, falling=True):
    idx = np.where(y <= target)[0] if falling else np.where(y >= target)[0]
    if len(idx) == 0 or (idx[0] == 0):
        return math.nan if len(idx) == 0 else float(freq[0])
    i = idx[0]
    f0, f1 = math.log10(freq[i - 1]), math.log10(freq[i])
    frac = (y[i - 1] - target) / (y[i - 1] - y[i])
    return float(10 ** (f0 + frac * (f1 - f0)))


def extract_metrics(raw: RawOutput, family: FamilySpec) -> dict[str, float]:
    """Compute every family metric the available analyses support; NaN otherwise."""
    plan, R = raw.plan, raw.results
    m: dict[str, float] = {}

    ac = R.get("ac")
    if ac and ac.data:
        f, db, ph = ac.data["freq"], ac.data["db"], ac.data["phase_deg"]
        ref_db = _interp_db(f, db, plan.f_ref)
        # gains / passband level (all dB, read at f_ref / DC)
        for k in ("gain_db", "dc_gain_db", "diff_gain_db", "passband_gain_db"):
            if any(x.key == k for x in family.metrics):
                m[k] = ref_db
        # -3 dB corner from passband — same measurement for filter cutoff and amp BW
        corner = _logf_cross(f, db, ref_db - _3DB, falling=True)
        for k in ("bw_3db_hz", "cutoff_freq_hz"):
            if any(x.key == k for x in family.metrics):
                m[k] = corner
        # stopband attenuation at 10x the corner
        if any(x.key == "stopband_atten_db" for x in family.metrics):
            m["stopband_atten_db"] = (ref_db - _interp_db(f, db, 10 * corner)
                                      if corner == corner and 10 * corner <= f[-1]
                                      else math.nan)
        # unity-gain frequency (GBW) + phase margin
        fc0 = _logf_cross(f, db, 0.0, falling=True)
        if any(x.key == "gbw_hz" for x in family.metrics):
            m["gbw_hz"] = fc0
        if any(x.key == "phase_margin_deg" for x in family.metrics):
            if fc0 == fc0:
                ph_c = float(np.interp(math.log10(fc0), np.log10(f), ph))
                m["phase_margin_deg"] = 180.0 + (ph_c - ph[0])
            else:
                m["phase_margin_deg"] = math.nan
        # Q (band-pass): f0 / (f_high - f_low)
        if any(x.key == "q_factor" for x in family.metrics):
            peak = db.max(); ipk = int(db.argmax())
            fhi = _logf_cross(f[ipk:], db[ipk:], peak - _3DB, falling=True)
            flo_arr = db[:ipk + 1][::-1]; ff = f[:ipk + 1][::-1]
            flo = _logf_cross(ff, flo_arr, peak - _3DB, falling=True)
            m["q_factor"] = (f[ipk] / (fhi - flo)
                             if fhi == fhi and flo == flo and fhi > flo else math.nan)

    op = R.get("op")
    if op and op.data.get("branch") and any(x.key == "quiescent_power_w"
                                            for x in family.metrics):
        pq = sum(abs(v * cur) for (e, _, v), cur in
                 zip(plan.supplies, op.data["branch"].values()))
        m["quiescent_power_w"] = pq

    dc = R.get("dc")
    if dc and dc.data and any(x.key == "out_swing_vpp" for x in family.metrics):
        vin, vout = dc.data["vin"], dc.data["vout"]
        g = np.gradient(vout, vin); gpk = np.max(np.abs(g))
        reg = np.abs(g) >= plan.icmr_gain_frac * gpk
        m["out_swing_vpp"] = float(vout[reg].max() - vout[reg].min()) if reg.any() else math.nan

    cm = R.get("ac_cmrr")
    if cm and cm.data.get("dm") and cm.data.get("cm") and any(
            x.key == "cmrr_db" for x in family.metrics):
        adm = _interp_db(cm.data["dm"]["freq"], cm.data["dm"]["db"], plan.cmrr_fref)
        acm = _interp_db(cm.data["cm"]["freq"], cm.data["cm"]["db"], plan.cmrr_fref)
        m["cmrr_db"] = (adm - acm) if acm > -200 else 200.0

    ic = R.get("dc_icmr")
    if ic and len(ic.data.get("vcm", [])) and any(x.key == "icmr_v"
                                                  for x in family.metrics):
        vcm, adm = ic.data["vcm"], ic.data["adm"]
        reg = adm >= plan.icmr_gain_frac * adm.max()
        m["icmr_v"] = float(vcm[reg].max() - vcm[reg].min()) if reg.any() else math.nan

    # metrics with no implemented recipe yet -> NaN (e.g. slew_rate)
    for x in family.metrics:
        m.setdefault(x.key, math.nan)
    return m


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
@dataclass
class ItemSpec:
    family: FamilySpec
    targets: dict                       # key -> float | (lo, hi) for WINDOW
    objectives: tuple[str, ...]         # active/graded metric keys
    tolerance: dict = field(default_factory=dict)   # key -> frac tol (TARGET)
    corner: str = "none"


def _to_lin(x, unit):
    return 10 ** (x / 20) if unit == "dB" else x


def _violation(metric, measured, target, tol):
    """max(0, log(target/measured)) family, in the metric's natural domain."""
    d, unit = metric.direction, metric.unit
    if isinstance(measured, float) and math.isnan(measured):
        return math.inf
    if d is Direction.WINDOW:                       # target = (lo, hi), dB band
        lo, hi = target
        return max(0.0, lo - measured, measured - hi) * DB_TO_LN
    mt, tt = _to_lin(measured, unit), _to_lin(target, unit)
    if d is Direction.GE:
        if mt <= 0:
            return math.inf
        return max(0.0, math.log(tt / mt))
    if d is Direction.LE:
        if mt <= 0:
            return 0.0
        return max(0.0, math.log(mt / tt))
    # TARGET (two-sided, within tol)
    if mt <= 0:
        return math.inf
    return max(0.0, abs(math.log(mt / tt)) - math.log(1 + tol))


def _passes(metric, measured, target, tol):
    if isinstance(measured, float) and math.isnan(measured):
        return False
    d = metric.direction
    if d is Direction.GE:
        return measured >= target
    if d is Direction.LE:
        return measured <= target
    if d is Direction.WINDOW:
        return target[0] <= measured <= target[1]
    return abs(math.log(_to_lin(measured, metric.unit) /
                        _to_lin(target, metric.unit))) <= math.log(1 + tol)


# graded = syntax bonus + convergence bonus + gate-gated metric quality
W_SYNTAX, W_CONV, W_METRICS = 0.1, 0.1, 0.8


def score(metrics: dict, spec: ItemSpec, *, syntax_ok: bool = True,
          converged: bool = True, gates_pass: bool = True) -> dict:
    fam = spec.family
    per_pass, quality, viol = {}, {}, {}
    for key in spec.objectives:
        mspec = fam.metric(key)
        target = spec.targets[key]
        tol = spec.tolerance.get(key, mspec.base_tol)
        measured = metrics.get(key, math.nan)
        v = _violation(mspec, measured, target, tol)
        viol[key] = v
        quality[key] = math.exp(-v) if v != math.inf else 0.0
        per_pass[key] = _passes(mspec, measured, target, tol)

    metric_q = (sum(quality.values()) / len(quality)) if quality else 0.0
    metric_term = metric_q * (1.0 if gates_pass else 0.0)
    graded = (W_SYNTAX * float(syntax_ok)
              + W_CONV * float(converged and syntax_ok)
              + W_METRICS * (metric_term if converged and syntax_ok else 0.0))
    return {
        "per_metric_pass": per_pass,
        "graded": graded,
        "all_pass": bool(per_pass) and all(per_pass.values()) and gates_pass,
        "metric_quality": quality,
        "violations": viol,
        "gates_pass": gates_pass,
        "syntax_ok": syntax_ok,
        "converged": converged,
    }


# --------------------------------------------------------------------------- #
# failure-mode classifier (deterministic label per graded response)
# --------------------------------------------------------------------------- #
class FailureMode(str, Enum):
    PASS = "pass"
    PARSE_FAILURE = "parse_failure"
    NON_CONVERGENCE = "non_convergence"
    WRONG_TOPOLOGY = "wrong_topology"
    MIS_SIZED = "topology_correct_mis_sized"
    NEAR_MISS = "single_constraint_near_miss"
    MULTI_OBJECTIVE_FAILURE = "multi_objective_failure"


# a single failed constraint with violation <= this (natural-log-ratio units,
# ~16% past the pass boundary; ~1.3 dB for dB metrics) is a near-miss, not a
# gross sizing error.
NEAR_MISS_EPS = 0.15

# terminal counts for the topology fingerprint (naming-independent structure)
_NTERMS = {"M": 4, "J": 4, "E": 4, "G": 4, "T": 4, "Q": 3,
           "R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "D": 2, "B": 2, "H": 2, "F": 2}


@dataclass(frozen=True)
class Fingerprint:
    types: tuple            # sorted ((element_type, count), ...)
    n_nodes: int
    degree_seq: tuple       # sorted node degrees (ignores node names)


def fingerprint(netlist: str) -> Fingerprint:
    """Structural signature of the respondent circuit: element-type histogram +
    node-degree sequence. Ignores comments, dot-cards, values, and node names."""
    types: dict[str, int] = {}
    deg: dict[str, int] = {}
    for raw in netlist.splitlines():
        line = raw.strip()
        if not line or line[0] in "*." :
            continue
        tok = line.split()
        t = tok[0][0].upper()
        if t not in _NTERMS:
            continue
        types[t] = types.get(t, 0) + 1
        for n in tok[1:1 + _NTERMS[t]]:
            deg[n] = deg.get(n, 0) + 1
    return Fingerprint(tuple(sorted(types.items())), len(deg),
                       tuple(sorted(deg.values())))


def topology_match(fp: Fingerprint, ref: Fingerprint) -> bool:
    """Same element-type histogram AND same degree sequence ⇒ same topology
    (values may differ). A missing/extra device or different wiring breaks it."""
    return fp.types == ref.types and fp.degree_seq == ref.degree_seq


def classify(netlist: str, raw: RawOutput, score_result: dict,
             reference_fp: Fingerprint | None = None) -> FailureMode:
    """Deterministic failure label, in priority order. Doubles as a debug log.

    Note PASS is checked before WRONG_TOPOLOGY: the grader is functional, so a
    design that meets every spec is a pass even if its topology differs from our
    (single) reference solution. WRONG_TOPOLOGY is reserved for designs that
    *fail* AND don't match the reference shape — "you built the wrong circuit."
    """
    if not raw.syntax_ok:
        return FailureMode.PARSE_FAILURE
    if not raw.converged:
        return FailureMode.NON_CONVERGENCE
    if not score_result.get("gates_pass", True):
        return FailureMode.MIS_SIZED          # right shape, but biased/sized so it
                                              # doesn't even operate (gate failure)
    fails = [k for k, ok in score_result["per_metric_pass"].items() if not ok]
    if not fails:
        return FailureMode.PASS               # meets spec -> pass, topology aside
    if reference_fp is not None and not topology_match(fingerprint(netlist), reference_fp):
        return FailureMode.WRONG_TOPOLOGY     # fails AND wrong shape
    if len(fails) == 1:
        v = score_result["violations"][fails[0]]
        return (FailureMode.NEAR_MISS if v <= NEAR_MISS_EPS
                else FailureMode.MIS_SIZED)
    return FailureMode.MULTI_OBJECTIVE_FAILURE


# --------------------------------------------------------------------------- #
# smoke test: simulate -> extract -> score end-to-end (model-free)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from circuit_irt.families import FILTERS, CS_AMP, DIFF_PAIR, TWO_STAGE_OPAMP

    print("=== 1. RC low-pass filter (ac: gain/BW/stopband) ===")
    rc = "R1 in out 1k\nC1 out 0 159.155n"
    plan = AnalysisPlan(("ac",), in_nodes=("in",), ac=(1, 1e6, "dec", 50), f_ref=10)
    raw = simulate(rc, plan)
    mm = extract_metrics(raw, FILTERS)
    item = ItemSpec(FILTERS,
                    targets={"cutoff_freq_hz": 1000.0, "passband_gain_db": (-1.0, 0.0),
                             "stopband_atten_db": 15.0},
                    objectives=("cutoff_freq_hz", "passband_gain_db", "stopband_atten_db"))
    s = score(mm, item, syntax_ok=raw.syntax_ok, converged=raw.converged)
    print(f"  status={raw.status.value}  fc={mm['cutoff_freq_hz']:.1f}Hz "
          f"atten={mm['stopband_atten_db']:.1f}dB  pass={s['per_metric_pass']} "
          f"graded={s['graded']:.3f}")
    assert s["all_pass"] and s["graded"] > 0.99

    print("=== 2. quiescent power (op) ===")
    rail = "Rload vdd 0 1k"
    p2 = AnalysisPlan(("op",), supplies=(("VDD", "vdd", 5.0),))
    raw2 = simulate(rail, p2); mm2 = extract_metrics(raw2, CS_AMP)
    it2 = ItemSpec(CS_AMP, {"quiescent_power_w": 30e-3}, ("quiescent_power_w",))
    s2 = score(mm2, it2, syntax_ok=raw2.syntax_ok, converged=raw2.converged)
    print(f"  Pq={mm2['quiescent_power_w']*1e3:.2f}mW pass={s2['per_metric_pass']} "
          f"graded={s2['graded']:.3f}")
    assert s2["all_pass"]

    print("=== 3. behavioral 2-stage op-amp (ac: A0/GBW/PM) ===")
    A0, p1, p2f = 1000.0, 1e3, 3e6
    C1 = 1 / (2 * math.pi * 1e3 * p1); C2 = 1 / (2 * math.pi * 1e3 * p2f)
    op = (f"E1 n1 0 inp inm {A0}\nR1 n1 n2 1k\nC1 n2 0 {C1:.6e}\n"
          f"E2 n3 0 n2 0 1\nR2 n3 out 1k\nC2 out 0 {C2:.6e}")
    p3 = AnalysisPlan(("ac",), in_nodes=("inp", "inm"), input_mode="differential",
                      ac=(1, 1e8, "dec", 50), f_ref=10)
    raw3 = simulate(op, p3); mm3 = extract_metrics(raw3, TWO_STAGE_OPAMP)
    it3 = ItemSpec(TWO_STAGE_OPAMP,
                   {"dc_gain_db": 55.0, "gbw_hz": 5e5, "phase_margin_deg": 45.0},
                   ("dc_gain_db", "gbw_hz", "phase_margin_deg"))
    s3 = score(mm3, it3, syntax_ok=raw3.syntax_ok, converged=raw3.converged)
    print(f"  A0={mm3['dc_gain_db']:.1f}dB GBW={mm3['gbw_hz']:.3g}Hz "
          f"PM={mm3['phase_margin_deg']:.1f}deg pass={s3['per_metric_pass']} "
          f"graded={s3['graded']:.3f}")
    assert s3["per_metric_pass"]["dc_gain_db"] and s3["per_metric_pass"]["phase_margin_deg"]

    print("=== 4. behavioral diff pair (ac_cmrr + dc_icmr) ===")
    vc = "((v(inp)+v(inm))/2)"
    win = f"0.25*(1+tanh(({vc}-0.3)/0.03))*(1+tanh((1.5-{vc})/0.03))"
    dp = f"B1 out 0 V = 100*{win}*(v(inp)-v(inm)) + 0.1*{vc}"
    p4 = AnalysisPlan(("ac_cmrr", "dc_icmr"), in_nodes=("inp", "inm"),
                      input_mode="differential", input_bias=0.9,
                      ac=(10, 1e3, "dec", 5), cmrr_fref=100, icmr=(0.0, 1.8, 0.05))
    raw4 = simulate(dp, p4); mm4 = extract_metrics(raw4, DIFF_PAIR)
    it4 = ItemSpec(DIFF_PAIR, {"cmrr_db": 40.0, "icmr_v": 0.8}, ("cmrr_db", "icmr_v"))
    s4 = score(mm4, it4, syntax_ok=raw4.syntax_ok, converged=raw4.converged)
    print(f"  CMRR={mm4['cmrr_db']:.1f}dB ICMR={mm4['icmr_v']:.2f}V "
          f"pass={s4['per_metric_pass']} graded={s4['graded']:.3f}")
    assert s4["all_pass"]

    print("=== 5. convergence-failure capture ===")
    raw5 = simulate("V1 a 0 DC 1\nV2 a 0 DC 2", AnalysisPlan(("op",)))
    print(f"  voltage-source loop -> status={raw5.status.value} "
          f"converged={raw5.converged}")
    raw6 = simulate("Xbogus a b c", AnalysisPlan(("op",)))
    print(f"  garbage element -> status={raw6.status.value} syntax_ok={raw6.syntax_ok}")
    s6 = score({}, ItemSpec(FILTERS, {"cutoff_freq_hz": 1e3}, ("cutoff_freq_hz",)),
               syntax_ok=raw6.syntax_ok, converged=raw6.converged)
    print(f"  graded(parse-fail)={s6['graded']:.3f}")
    assert not raw5.converged and raw6.status is SimStatus.SYNTAX_ERROR
    assert s6["graded"] == 0.0

    print("\nOK: simulate -> extract_metrics -> score validated end-to-end.")
