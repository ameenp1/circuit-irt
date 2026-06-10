# The four families — locked spec schemas (Week 2)

Human-readable view of `families.py` (the canonical machine-readable source). Each
family fixes: the topology, the design variables the respondent sets, the test
conditions, the **metrics + valid ranges** (the spec schema), and the **difficulty
axes**. Measurement recipes live in [`spec_glossary.md`](spec_glossary.md) (§ refs).

## Difficulty axes (common encoding)

Every family varies along the three axes from the proposal:

| axis | meaning | encoded as |
|---|---|---|
| **constraint-tightness** `t` | how little slack a target allows | TARGET tol = `base_tol/t`; GE/LE targets pushed into the hard end of `sample_range` as `t↑`. Levels `(1, 2, 4, 8)` |
| **# simultaneous objectives** | how many constraints must hold at once | sample a subset of `objective_pool`, size in `[min, max]` |
| **robustness / corner** | must meet spec across a corner, not just nominal | one of `corner_options` (always includes `none`) |

Targets are sampled from each metric's **valid range**; a concrete item is a family
with targets + active objectives + a corner filled in. Reference solutions
(Week 3) must pass the item through the harness before it enters the bank.

---

## 1. RC / RLC filters  (`filters`)

Passive 1st/2nd-order LP/HP/BP. Nodes `in / out / 0`. Design vars: **R, C, (L)**.
Test: `V(in) AC 1`, AC sweep 1 Hz–100 MHz (dec 50).

| metric | dir | valid range | unit | recipe |
|---|---|---|---|---|
| passband gain | window | −1 … 0 | dB | §1 |
| cutoff fc (LP/HP) / f0 (BP) | target | 1e2 … 1e5 | Hz | §2 |
| stopband atten @10×fc | ≥ | 15 … 40 | dB | §2 |
| Q factor (BP only) | target | 0.5 … 5 | — | §2 |

Objectives: 1–3 of {fc, stopband, passband gain, Q}. Corners: `none / comp_tol_5pct /
comp_tol_10pct` (fc must hold under ±5–10 % R,C).

## 2. Common-source amplifier  (`cs_amp`)

Single MOSFET + load + bias. Nodes `in / out / vdd / 0` (VDD 1.8 V). Design vars:
**M1 W/L, RD, bias, caps**. Test: `.op` (power) · AC 1 Hz–1 GHz (gain/BW) · DC sweep
(swing). **Gate:** M1 in saturation.

| metric | dir | valid range | unit | recipe |
|---|---|---|---|---|
| midband gain | ≥ | 15 … 35 | dB | §1 |
| upper −3 dB BW | target | 1e5 … 1e8 | Hz | §2 |
| output swing | ≥ | 0.4 … 1.2 | Vpp | §4 |
| quiescent power | ≤ | 50 µ … 2 m | W | §5 |

Objectives: 1–4 of {gain, BW, swing, power}. Corners: `none / supply_pm10 /
temp_0_85C / vth_corner`.

## 3. Differential pair  (`diff_pair`)

Matched pair + tail source + load; **single-ended output** (finite CMRR). Nodes
`inp / inm / out / vdd / 0 / tail` (VDD 1.8 V, Vcm 0.9 V). Design vars: **M1/M2 W/L,
tail current, load**. Test: `.op` · DM+CM AC for CMRR (`f_ref` 1 kHz) · Vcm sweep
0–1.8 V for ICMR. **Gates:** M1/M2 + tail in saturation.

| metric | dir | valid range | unit | recipe |
|---|---|---|---|---|
| differential gain A_dm | ≥ | 15 … 40 | dB | §1 |
| **CMRR** | ≥ | 40 … 80 | dB | **§6 (new)** |
| **ICMR** | ≥ | 0.4 … 1.2 | V | **§7 (new)** |
| quiescent power | ≤ | 50 µ … 2 m | W | §5 |
| differential −3 dB BW | target | 1e5 … 5e7 | Hz | §2 |

Objectives: 2–5 of {A_dm, CMRR, ICMR, power, BW}. Corners: `none / tail_finite_ro /
mismatch_1pct / supply_pm10` (the finite-tail-`ro` and mismatch corners are where
CMRR genuinely degrades — the family's signature hard axis).

## 4. Two-stage op-amp  (`two_stage_opamp`)

Diff-pair stage + CS stage + Miller compensation, open-loop into a specified **CL
(5 pF)**. Nodes `inp / inm / out / vdd / 0` (VDD 1.8 V). Design vars: **pair sizes,
bias current, 2nd-stage device, Miller Cc (+ Rz), CL**. Test: open-loop AC (A0/GBW/PM)
· `.op` (power) · DC (swing) · tran (slew, optional). **Gate:** all devices saturated.

| metric | dir | valid range | unit | recipe |
|---|---|---|---|---|
| open-loop DC gain A0 | ≥ | 60 … 100 | dB | §1 |
| gain-bandwidth GBW | ≥ | 1e6 … 5e7 | Hz | §2/§3 |
| phase margin | ≥ | 45 … 65 | deg | §3 |
| quiescent power | ≤ | 100 µ … 5 m | W | §5 |
| output swing | ≥ | 0.6 … 1.4 | Vpp | §4 |
| slew rate (opt.) | ≥ | 1e5 … 1e7 | V/s | §4 tran (recipe added W3) |

Objectives: 2–6 of {A0, GBW, PM, power, swing, slew}. Corners: `none / cload_1_10pf /
supply_pm10 / temp_0_85C` (PM-holds-across-CL is the classic compensation-robustness
corner).

---

## Notes / open items for Week 3

- **Slew-rate** (`tran`) recipe is not yet in the glossary — add it with the Week 3
  generators (optional objective; doesn't block the rest).
- **Validity gates** (`gates`) are hard pass/fail bias checks (device saturation, tail
  compliance) enforced *before* graded metrics — implemented via the §7 region check
  (`@m[vds]` ≥ `@m[vdsat]`). Wire these into `score()` as a gate in Week 2.
- The `cmrr_db` metric uses analysis `ac_cmrr` (two excitations) and `icmr_v` uses
  `dc_icmr` (Python Vcm loop) — both need the diff-pair netlist to expose
  parameterised input sources, which the Week 3 generator must emit.
