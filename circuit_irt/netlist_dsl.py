"""Minimal netlist DSL + ngspice batch runner.

Programmatically builds SPICE netlists and runs them through `ngspice -b`,
then reads the results back. Deliberately small; this is the seed for the
Week 2 `simulate()` wrapper and the Week 3 spec-template generators.

Design choices:
  * Netlists are plain SPICE text emitted by a `Circuit` object — no PySpice /
    libngspice dependency. Simulation goes through `ngspice -b`, matching the
    harness the project standardizes on.
  * A `.control` block drives the analysis and uses `wrdata` to write results
    to a file, so output is read back as a plain numeric table.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class NgspiceError(RuntimeError):
    """Raised when ngspice is missing, times out, or reports a hard error."""


@dataclass
class Circuit:
    """Accumulates SPICE element lines + control statements into a netlist."""

    title: str = "circuit"
    _elements: list[str] = field(default_factory=list)
    _control: list[str] = field(default_factory=list)

    # --- element helpers (extend as families grow) ---------------------------
    def V(self, name: str, pos: str, neg: str, *, dc: float = 0.0,
          ac: float | None = None) -> "Circuit":
        spec = f"DC {dc}"
        if ac is not None:
            spec += f" AC {ac}"
        self._elements.append(f"V{name} {pos} {neg} {spec}")
        return self

    def R(self, name: str, n1: str, n2: str, value: str | float) -> "Circuit":
        self._elements.append(f"R{name} {n1} {n2} {value}")
        return self

    def C(self, name: str, n1: str, n2: str, value: str | float) -> "Circuit":
        self._elements.append(f"C{name} {n1} {n2} {value}")
        return self

    def L(self, name: str, n1: str, n2: str, value: str | float) -> "Circuit":
        self._elements.append(f"L{name} {n1} {n2} {value}")
        return self

    # --- control / analysis --------------------------------------------------
    def control(self, *lines: str) -> "Circuit":
        self._control.extend(lines)
        return self

    def to_netlist(self) -> str:
        lines = [f"* {self.title}", *self._elements]
        if self._control:
            lines += [".control", *self._control, ".endc"]
        lines.append(".end")
        return "\n".join(lines) + "\n"


def run_batch(netlist: str, *, timeout: float = 60.0,
              workdir: Path | None = None) -> tuple[str, Path]:
    """Run a netlist through `ngspice -b`. Returns (stdout, workdir).

    The netlist's `.control` block is expected to `wrdata` results into files
    relative to the working directory.
    """
    exe = shutil.which("ngspice")
    if exe is None:
        raise NgspiceError("ngspice not found on PATH")

    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ngspice_"))
    wd.mkdir(parents=True, exist_ok=True)
    cir = wd / "deck.cir"
    cir.write_text(netlist)

    try:
        proc = subprocess.run(
            [exe, "-b", cir.name],
            cwd=wd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise NgspiceError(f"ngspice timed out after {timeout}s") from e

    # ngspice prints warnings to stderr even on success; treat nonzero rc OR an
    # explicit fatal/convergence marker as failure (Week 2 hardens this further).
    blob = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or "fatal" in blob.lower():
        raise NgspiceError(
            f"ngspice failed (rc={proc.returncode}):\n{blob[-2000:]}"
        )
    return proc.stdout, wd


def read_wrdata(path: Path, ncols: int) -> np.ndarray:
    """Read an ngspice `wrdata` table into an (nrows, ncols) float array.

    `wrdata` repeats the scale column for every vector, so a deck writing
    vectors `a b` over frequency yields columns [freq, a, freq, b].
    """
    arr = np.loadtxt(path)
    if arr.ndim == 1:                      # single row
        arr = arr.reshape(1, -1)
    if arr.shape[1] != ncols:
        raise NgspiceError(
            f"expected {ncols} columns in {path.name}, got {arr.shape[1]}"
        )
    return arr
