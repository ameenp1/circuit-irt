"""Validate the spec-glossary measurement recipes against synthetic circuits
with closed-form answers. Confirms the ngspice-46 idioms work and seeds the
Week 2 `extract_metrics` implementation.

Run:  .venv/bin/python test_spec_recipes.py
"""
from __future__ import annotations

import math

import numpy as np

from circuit_irt.netlist_dsl import Circuit, read_wrdata, run_batch


# --- generic extractors (mirror docs/spec_glossary.md) -----------------------
def _logf_cross(freq, y, target, falling=True):
    """First crossing of y=target, interpolated linearly in log10(f)."""
    idx = np.where(y <= target)[0] if falling else np.where(y >= target)[0]
    if len(idx) == 0:
        return math.nan
    i = idx[0]
    if i == 0:
        return float(freq[0])
    f0, f1 = math.log10(freq[i - 1]), math.log10(freq[i])
    y0, y1 = y[i - 1], y[i]
    frac = (y0 - target) / (y0 - y1)
    return float(10 ** (f0 + frac * (f1 - f0)))


def ac_sweep(circuit_body, fstart, fstop, dec=50):
    """Run an AC sweep writing db and continuous-phase(deg) of h=v(out)/v(in)."""
    c = Circuit(title="ac")
    for fn in circuit_body:
        fn(c)
    c.control(
        f"ac dec {dec} {fstart} {fstop}",
        "let h = v(out)/v(in)",
        "let hd = db(h)",
        "let hp = 180*cph(h)/pi",        # continuous (unwrapped) phase, degrees
        "wrdata ac.txt hd hp",
    )
    _, wd = run_batch(c.to_netlist())
    d = read_wrdata(wd / "ac.txt", ncols=4)   # freq, hd, freq, hp
    return d[:, 0], d[:, 1], d[:, 3]


# --- tests -------------------------------------------------------------------
def test_gain_and_bw():
    R, C = 1_000.0, 159.155e-9                 # fc = 1/(2 pi R C) ~= 1000 Hz
    fc = 1.0 / (2 * math.pi * R * C)
    body = [
        lambda c: c.V("1", "in", "0", ac=1.0),
        lambda c: c.R("1", "in", "out", R),
        lambda c: c.C("1", "out", "0", C),
    ]
    f, hd, hp = ac_sweep(body, 1, 1e6)
    gain_db = float(np.interp(math.log10(10.0), np.log10(f), hd))  # f_ref = 10 Hz
    bw = _logf_cross(f, hd, hd[0] - 3.0103, falling=True)
    print(f"[gain/BW] gain@10Hz={gain_db:+.4f} dB (exp 0)  "
          f"BW={bw:.2f} Hz (exp {fc:.2f})")
    assert abs(gain_db) < 1e-2
    assert abs(bw - fc) / fc < 0.02


def test_phase_margin():
    # synthetic 2-pole amp: DC gain 1000 (60 dB), poles at 1k and 10k, ideal-buffered
    p1, p2 = 1_000.0, 10_000.0
    R1 = R2 = 1_000.0
    C1 = 1.0 / (2 * math.pi * R1 * p1)
    C2 = 1.0 / (2 * math.pi * R2 * p2)
    body = [
        lambda c: c.V("1", "in", "0", ac=1.0),
        lambda c: c._elements.append("E1 n1 0 in 0 1000"),   # VCVS gain 1000
        lambda c: c.R("1", "n1", "n2", R1),
        lambda c: c.C("1", "n2", "0", C1),
        lambda c: c._elements.append("E2 n3 0 n2 0 1"),       # ideal buffer
        lambda c: c.R("2", "n3", "out", R2),
        lambda c: c.C("2", "out", "0", C2),
    ]
    f, hd, hp = ac_sweep(body, 1, 1e7)
    fc = _logf_cross(f, hd, 0.0, falling=True)                # unity-gain crossover
    ph_c = float(np.interp(math.log10(fc), np.log10(f), hp))
    ph_low = hp[0]
    pm = 180.0 + (ph_c - ph_low)
    # closed form: PM = 180 - atan(fc/p1) - atan(fc/p2)
    pm_exp = 180 - math.degrees(math.atan(fc / p1)) - math.degrees(math.atan(fc / p2))
    print(f"[phase margin] fc={fc:.1f} Hz  PM={pm:.2f} deg (exp {pm_exp:.2f})")
    assert abs(pm - pm_exp) < 1.0


def test_output_swing():
    # behavioral clipping amp: Vout = 5*tanh(Vin); gain g=5*sech^2(Vin), g(0)=5.
    # knee at |g|=0.5*g_peak -> cosh^2=2 -> Vin=+-0.8814 -> Vout=+-3.536
    body_dc = Circuit(title="swing")
    body_dc.V("in", "in", "0", dc=0.0)
    body_dc._elements.append("B1 out 0 V = 5*tanh(v(in))")
    body_dc.control("dc Vin -5 5 0.01", "wrdata dc.txt v(out)")
    _, wd = run_batch(body_dc.to_netlist())
    d = read_wrdata(wd / "dc.txt", ncols=2)         # vin, vout
    vin, vout = d[:, 0], d[:, 1]
    g = np.gradient(vout, vin)
    gpeak = np.max(np.abs(g))
    region = np.abs(g) >= 0.5 * gpeak
    vhi, vlo = vout[region].max(), vout[region].min()
    print(f"[swing] Vout_high={vhi:.3f} Vout_low={vlo:.3f} pp={vhi-vlo:.3f} "
          f"(exp +-3.536, pp 7.07)")
    assert abs(vhi - 3.536) < 0.05 and abs(vlo + 3.536) < 0.05


def test_quiescent_power():
    # Vdd=5V across R=1k -> I=5mA, Pq = 25 mW
    c = Circuit(title="pq")
    c.V("dd", "vdd", "0", dc=5.0)
    c.R("load", "vdd", "0", 1_000.0)
    c.control("op", "wrdata op.txt vdd#branch")
    _, wd = run_batch(c.to_netlist())
    d = read_wrdata(wd / "op.txt", ncols=2)         # (dummy scale), branch current
    i_branch = d[0, 1]
    pq = abs(5.0 * i_branch)
    print(f"[quiescent power] i(vdd)={i_branch*1e3:.3f} mA  Pq={pq*1e3:.3f} mW (exp 25)")
    assert abs(pq - 0.025) < 1e-4


def test_cmrr():
    # linear behavioral diff amp: out = 100*(inp-inm) + 0.1*((inp+inm)/2)
    #   -> A_dm=100 (40 dB), A_cm=0.1 (-20 dB) -> CMRR = 40-(-20) = 60 dB.
    # Validates the DM/CM excitation + extraction method, free of device physics.
    def run(ac_p, ph_p, ac_m, ph_m):
        c = Circuit(title="cmrr")
        c._elements.append(f"Vp inp 0 DC 0 AC {ac_p} {ph_p}")
        c._elements.append(f"Vm inm 0 DC 0 AC {ac_m} {ph_m}")
        c._elements.append("B1 out 0 V = 100*(v(inp)-v(inm)) + 0.1*((v(inp)+v(inm))/2)")
        c.control("ac dec 5 10 1e3", "let g = db(v(out))", "wrdata cmrr.txt g")
        _, wd = run_batch(c.to_netlist())
        return read_wrdata(wd / "cmrr.txt", ncols=2)[0, 1]   # db at f_ref (flat)
    adm_db = run(0.5, 0, 0.5, 180)        # v_id=1, v_ic=0
    acm_db = run(1.0, 0, 1.0, 0)          # v_ic=1, v_id=0
    cmrr = adm_db - acm_db
    print(f"[CMRR] A_dm={adm_db:.2f} dB  A_cm={acm_db:.2f} dB  CMRR={cmrr:.2f} dB (exp 60)")
    assert abs(adm_db - 40.0) < 0.1 and abs(acm_db + 20.0) < 0.1
    assert abs(cmrr - 60.0) < 0.1


def test_icmr():
    # behavioral pair whose differential gain is windowed in Vcm:
    #   A_dm(Vcm) = 100 * w(Vcm),  w = 0.25*(1+tanh((Vcm-Vlo)/s))*(1+tanh((Vhi-Vcm)/s))
    # w = 0.5 exactly at Vcm = Vlo and Vhi, so the 0.5*peak gain-fraction edges
    # recover the set window. Validates the Vcm-sweep + knee-extraction mechanics.
    Vlo, Vhi, s, A0 = -1.0, 1.0, 0.03, 100.0
    vcms = np.arange(-1.8, 1.8001, 0.05)
    gains = []
    for vcm in vcms:
        c = Circuit(title="icmr")
        c._elements.append(f"Vp inp 0 DC {vcm} AC 0.5 0")
        c._elements.append(f"Vm inm 0 DC {vcm} AC 0.5 180")     # v_id = 1
        vc = "((v(inp)+v(inm))/2)"
        w = f"0.25*(1+tanh(({vc}-({Vlo}))/{s}))*(1+tanh((({Vhi})-{vc})/{s}))"
        c._elements.append(f"B1 out 0 V = {A0}*{w}*(v(inp)-v(inm))")
        c.control("ac dec 2 1e3 1e4", "let g = mag(v(out))", "wrdata icmr.txt g")
        _, wd = run_batch(c.to_netlist())
        gains.append(read_wrdata(wd / "icmr.txt", ncols=2)[0, 1])
    gains = np.array(gains)
    region = gains >= 0.5 * gains.max()           # icmr_gain_frac = 0.5
    vcm_low, vcm_high = vcms[region].min(), vcms[region].max()
    print(f"[ICMR] Vcm_low={vcm_low:.2f} Vcm_high={vcm_high:.2f} "
          f"icmr={vcm_high-vcm_low:.2f} V (exp [-1.0, 1.0], 2.0)")
    assert abs(vcm_low - Vlo) < 0.06 and abs(vcm_high - Vhi) < 0.06


if __name__ == "__main__":
    test_gain_and_bw()
    test_phase_margin()
    test_output_swing()
    test_quiescent_power()
    test_cmrr()
    test_icmr()
    print("\nOK: all seven glossary recipes validated against closed-form values.")
