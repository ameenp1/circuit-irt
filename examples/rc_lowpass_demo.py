"""Programmatically generate an RC low-pass, AC-sweep it through ngspice,
read the result back, and confirm the measured -3 dB corner matches theory.

Run:  .venv/bin/python rc_lowpass_demo.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from circuit_irt.netlist_dsl import Circuit, read_wrdata, run_batch


def build_rc_lowpass(r_ohms: float, c_farads: float,
                     f_start: float = 1.0, f_stop: float = 1e6,
                     points_per_dec: int = 50) -> Circuit:
    """RC low-pass: V1 -> R -> (out) -> C -> gnd, driven by a 1 V AC source."""
    c = Circuit(title=f"RC low-pass R={r_ohms} C={c_farads}")
    c.V("1", "in", "0", ac=1.0)
    c.R("1", "in", "out", r_ohms)
    c.C("1", "out", "0", c_farads)
    c.control(
        f"ac dec {points_per_dec} {f_start} {f_stop}",
        "let vm = mag(v(out))",          # |H(f)| as a real vector
        "let vp = ph(v(out))",           # phase (radians)
        "wrdata ac_out.txt vm vp",       # -> cols: freq, vm, freq, vp
    )
    return c


def measured_minus3db(freq: np.ndarray, mag: np.ndarray) -> float:
    """Interpolate the frequency where |H| crosses -3 dB of its DC value."""
    dc = mag[0]
    target = dc / math.sqrt(2.0)         # -3 dB
    below = np.where(mag <= target)[0]
    if len(below) == 0:
        raise ValueError("magnitude never crossed -3 dB within the sweep")
    i = below[0]
    if i == 0:
        return float(freq[0])
    # log-log linear interpolation between the bracketing points
    f0, f1 = math.log10(freq[i - 1]), math.log10(freq[i])
    m0, m1 = mag[i - 1], mag[i]
    frac = (m0 - target) / (m0 - m1)
    return float(10 ** (f0 + frac * (f1 - f0)))


def main() -> None:
    R, C = 1_000.0, 159.155e-9           # fc = 1/(2*pi*R*C) ~= 1000 Hz
    fc_theory = 1.0 / (2 * math.pi * R * C)

    circuit = build_rc_lowpass(R, C)
    print("=== generated netlist ===")
    print(circuit.to_netlist())

    stdout, wd = run_batch(circuit.to_netlist())
    out_file = wd / "ac_out.txt"
    print(f"ngspice wrote: {out_file}  ({out_file.stat().st_size} bytes)")

    # read the output back
    data = read_wrdata(out_file, ncols=4)        # freq, vm, freq, vp
    freq, vm, vp = data[:, 0], data[:, 1], data[:, 3]
    print(f"read back {len(freq)} rows; "
          f"freq {freq[0]:.3g}..{freq[-1]:.3g} Hz, "
          f"|H| {vm.max():.4f}..{vm.min():.4g}")

    fc_meas = measured_minus3db(freq, vm)
    err_pct = 100 * abs(fc_meas - fc_theory) / fc_theory
    phase_at_fc = math.degrees(np.interp(fc_meas, freq, vp))

    print("\n=== results ===")
    print(f"DC gain |H(f0)|      : {vm[0]:.4f}      (expect ~1.000)")
    print(f"fc theoretical       : {fc_theory:8.2f} Hz")
    print(f"fc measured (-3 dB)  : {fc_meas:8.2f} Hz")
    print(f"error                : {err_pct:.2f} %")
    print(f"phase at fc          : {phase_at_fc:.2f} deg (expect ~-45)")

    # sanity assertions — this is the "confirm you can read it back" check
    assert abs(vm[0] - 1.0) < 1e-3, "DC gain should be ~1"
    assert err_pct < 2.0, f"fc off by {err_pct:.2f}% (>2%)"
    assert abs(phase_at_fc + 45.0) < 3.0, "phase at fc should be ~-45 deg"
    print("\nOK: RC low-pass round-tripped through ngspice and verified.")


if __name__ == "__main__":
    main()
