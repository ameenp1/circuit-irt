# Metric spec glossary

The exact definition + `ngspice -b` measurement recipe for each metric. This is
the contract for `extract_metrics(raw_output, family)` (Week 2). Each recipe is
phrased as: **analysis directive → vectors to `wrdata` → Python extraction rule →
edge cases**, consistent with `netlist_dsl.py` (`.control` block, `wrdata`,
`read_wrdata`).

> **Validated.** All seven recipes are checked against a synthetic circuit with a
> closed-form answer in `test_spec_recipes.py` (ngspice-46): gain/BW on an RC
> low-pass, phase margin on a 2-pole VCVS amp, output swing on a `tanh` clipper,
> quiescent power on a resistive rail, CMRR on a linear behavioral diff amp
> (recovers 60 dB), and ICMR on a Vcm-windowed behavioral pair (recovers the set
> window). Run `.venv/bin/python tests/test_spec_recipes.py`.

## Shared conventions (apply to every metric)

- **Nodes:** input `in`, output `out`, ground `0`. Supplies named per item
  (`vdd`, `vss`). The exact in/out/supply names are per-item spec parameters, not
  hard-coded.
- **AC drive:** the input source is `AC 1` (magnitude exactly 1). Then `v(out)` *is*
  the transfer function H(f). For robustness against accidental input scaling,
  prefer the ratio form `let h = v(out)/v(in)` and measure on `h`.
- **`wrdata` columns:** `wrdata` repeats the scale column per vector, so writing
  `a b` over frequency yields `[freq, a, freq, b]` → read column indices `0,1,3`
  (see `read_wrdata(path, ncols)`).
- **Phase:** degrees, and **unwrapped** before any crossing read (ngspice `cph()` is
  radians-continuous → `*180/pi`; or `numpy.unwrap` on radians then convert). Raw
  `vp()` wraps at ±180° and will corrupt phase-margin reads.
- **dB / −3 dB constant:** `3 dB = 20·log10(√2) = 3.0103`. `vdb(out) = 20·log10|H|`.
- **Frequency interpolation:** always interpolate crossings **linearly in log10(f)**.
- **Sweep density:** AC `dec 50` (≥20 minimum). The sweep must bracket the feature
  (corner / 0-dB crossover); if it doesn't, the extractor returns `NaN` + a reason
  flag — it must never crash or silently clamp (Week 4 hardening).
- **Convergence:** non-convergence / `ngspice` error → metric `NaN`, item fails for
  that reason (Week 2 captures this in `simulate`).

---

## 1. Gain — AC magnitude at a reference frequency

- **Definition:** `gain_dB = 20·log10 |H(f_ref)|`, `H = V(out)/V(in)`, at a per-item
  reference frequency `f_ref` (DC/low-band for amplifiers; passband center for
  filters). Linear gain `Av = |H(f_ref)|`.
- **Analysis:** `.ac dec 50 f_start f_stop` with `f_ref` inside `[f_start, f_stop]`.
- **Vectors:** `let h = v(out)/v(in)` → `wrdata ac.txt db(h) cph(h)` (mag dB, phase).
- **Extraction:** interpolate `db(h)` at `log10(f_ref)`. (If `f_ref` is "DC", use the
  lowest swept point and require `f_start ≤ f_ref/10`.)
- **Edge cases:** `f_ref` outside sweep → `NaN`+flag. Report both `gain_dB` and `Av`.

## 2. −3 dB bandwidth

- **Definition:** frequency where `|H|` falls to `1/√2` of its reference (passband)
  value, i.e. `db(h)` drops `3.0103 dB` below `gain_dB`. Low-pass: `BW = f_3dB` (upper
  corner). Band-pass: `BW = f_high − f_low` between the two −3 dB points.
- **Analysis:** same AC sweep as §1; must extend ≥1 decade past the expected corner.
- **Vectors:** `db(h)` (reuse §1 output).
- **Extraction:** `ref = gain_dB`; `target = ref − 3.0103`. Walk outward from `f_ref`;
  first sample with `db(h) ≤ target` on the falling side defines the bracket;
  log-f interpolate to the crossing. Band-pass: find the crossing on each side.
- **Edge cases:** no crossing within sweep (gain never drops 3 dB) → `NaN`+
  `"corner_outside_sweep"`; widen `f_stop` and re-run rather than report a wrong BW.

## 3. Phase margin — phase at unity-gain crossover

- **Definition:** `PM = 180° + [∠H(f_c) − ∠H(f_low)]`, where `f_c` is the unity-gain
  (0 dB, `|H|=1`) crossover and `∠H(f_low)` is the low-frequency phase. Subtracting
  `∠H(f_low)` references phase to DC so the formula is correct for both non-inverting
  (`∠H(f_low)≈0°`) and inverting (`≈±180°`) open-loop responses. Stable ⇔ `PM > 0`.
- **Analysis:** open-loop `.ac dec 50 f_start f_stop`; sweep must span DC-gain down
  through the 0-dB crossover.
- **Vectors:** `db(h)` and unwrapped phase `cph(h)*180/pi` (degrees).
- **Extraction:** find `f_c` = first **falling** crossing of `db(h)=0` (log-f
  interpolate); interpolate unwrapped phase at `f_c` → `ph_c`; `ph_low` = phase at
  `f_start`; `PM = 180 + (ph_c − ph_low)`.
- **Related (free from same sweep):** gain margin `GM = −db(h)` at the frequency
  where phase = `ph_low − 180°`.
- **Edge cases:** `db(h)` never crosses 0 within sweep → `NaN`+`"no_unity_crossing"`.
  Multiple crossings → take the lowest-frequency falling one. Phase **must** be
  unwrapped or `PM` can be off by 360°.

## 4. Output swing

- **Definition:** the output voltage range over which the circuit stays in its linear
  region — i.e. between the clipping knees where small-signal gain collapses. Report
  `Vout_high`, `Vout_low`, and `swing_pp = Vout_high − Vout_low`.
- **Analysis (primary, deterministic):** DC transfer sweep
  `.dc Vin v_lo v_hi v_step` across the full input range.
- **Vectors:** `wrdata dc.txt v(out)` (scale = `v-sweep`).
- **Extraction:** numerically differentiate `g(Vin) = d v(out)/d Vin`; let
  `g_peak = max|g|` near the high-gain center. The linear region is the contiguous
  span around the center where `|g| ≥ swing_gain_frac · g_peak`
  (`swing_gain_frac` ~ 0.5, a per-item spec parameter); `Vout_high/low` = `v(out)` at
  that region's edges. This locates the knees rather than assuming the rails.
- **Alternative (large-signal):** apply a large input sine, `.tran`, discard startup
  cycles, `swing_pp = max(v(out)) − min(v(out))` in steady state. Use when a real
  THD-bounded swing is wanted; note it adds timing/convergence risk.
- **Edge cases:** no clear gain peak (monotonic, no clipping in range) → swing is
  rail-limited; report the swept `v(out)` extrema + `"no_knee_in_range"` flag.

## 5. Quiescent power — from operating point

- **Definition:** static DC power with inputs at bias and no signal:
  `P_q = Σ_k |V_k · I(V_k)|` summed over all DC supply sources `V_k`
  (`I(V_k)` = supply branch current). For a single rail, `P_q = |Vdd · I(Vdd)|`.
- **Analysis:** `.op` (operating point).
- **Vectors:** supply branch currents. In ngspice the current through source `Vdd` is
  `i(vdd)` / vector `vdd#branch`. Recipe: `op` then
  `wrdata op.txt @vdd[i]` — or `print vdd#branch vss#branch` and parse stdout.
- **Extraction:** for each supply `V_k`, `P_k = |V_k_value · I_branch_k|`; sum.
  Take magnitudes so source sign convention (ngspice branch current flows + → −
  internally) can't flip the result; this gives power **drawn from the supplies**.
- **Edge cases:** forgetting a rail undercounts power → enumerate supplies from the
  item spec, not from guesswork. `.op` non-convergence → `NaN`+fail.

## 6. CMRR — common-mode rejection ratio (differential pair)

- **Definition:** `CMRR_dB = 20·log10|A_dm / A_cm| = gain_dm_dB − gain_cm_dB`, where
  `A_dm` is the differential-mode gain (output per unit `v_id = v(inp) − v(inm)`) and
  `A_cm` the common-mode gain (output per unit `v_ic = (v(inp)+v(inm))/2`). Output is
  single-ended `v(out)` or differential `v(outp) − v(outm)` per item.
- **Nodes:** differential inputs `inp`,`inm` biased at a DC common-mode `Vcm`; AC
  excitation sits on top of that bias.
- **Analysis:** **two AC runs** at a shared reference frequency `f_ref` (below the
  dominant pole, so both gains are flat):
  - **DM run:** `inp = AC 0.5 0`, `inm = AC 0.5 180` → `v_id=1, v_ic=0`. `A_dm = |H_out|`.
  - **CM run:** `inp = AC 1 0`, `inm = AC 1 0` → `v_ic=1, v_id=0`. `A_cm = |H_out|`.
- **Vectors:** each run writes `db(out_expr)` where `out_expr = v(out)` (single-ended)
  or `v(outp)-v(outm)` (differential).
- **Extraction:** `CMRR_dB = db(A_dm) − db(A_cm)` at `f_ref`.
- **Implementation:** the generator emits the diff-pair netlist with the two input
  sources parameterised; the `ac_cmrr` extractor renders/`alter`s the AC `mag/phase`
  of `inp`,`inm` for the two runs. (In-deck: `ac` → `alter @vinp[acmag]/[acphase]`,
  `alter @vinm[...]` → `ac` again.)
- **Edge cases:** a *balanced fully-differential* output gives `A_cm≈0` → `CMRR→∞`;
  cap at e.g. `+200 dB` and flag when `|A_cm| < 1e-9` (below numeric floor). Use the
  **single-ended** output for a finite, meaningful diff-pair CMRR. `f_ref` must be
  sub-dominant-pole or CMRR rolls off and the number is meaningless.

## 7. ICMR — input common-mode range (differential pair)

- **Definition:** the span of input common-mode voltage `Vcm` over which the pair
  stays functional (all devices saturated, differential gain maintained). Functional
  operationalisation (simulator-agnostic, mirrors §4 output swing): ICMR = the
  contiguous `Vcm` interval where `|A_dm(Vcm)| ≥ icmr_gain_frac · max_Vcm|A_dm|`
  (`icmr_gain_frac ≈ 0.5`, per-item parameter). Report `vcm_low`, `vcm_high`,
  `icmr = vcm_high − vcm_low`.
- **Analysis:** Python-driven sweep (matches the harness): for `Vcm` in
  `[v_lo … v_hi]` step `Δ` (span rail-to-rail, `v_lo=Vss`, `v_hi=Vdd`), set the input
  DC common-mode bias to `Vcm` and run a single-frequency **DM** `.ac` at `f_ref`
  (`inp=AC 0.5 0`, `inm=AC 0.5 180`); record `A_dm(Vcm) = |H_out|`.
- **Extraction:** find the contiguous window around the peak meeting the gain-fraction
  threshold; linear-interpolate the two edges in `Vcm`.
- **Rigorous cross-check:** at each `Vcm`, `.op` and test every device region via
  `@m1[vds] ≥ @m1[vdsat]` (saturation) plus tail-source compliance; ICMR = `Vcm` range
  where all hold. The gain-fraction method is primary; the region check is the
  physical validator.
- **Edge cases:** gain never drops within the swept range → `icmr ≥ span`, flag
  `"icmr_exceeds_sweep"` and widen. Multiple humps → take the region containing the
  global max.

---

## Per-item spec parameters (referenced above)

`in/out` node names (diff: `inp/inm/outp/outm`) · supply names + nominal values ·
`f_ref` · AC `f_start/f_stop` · DC `Vin` range/step · `swing_gain_frac` · `Vcm` bias +
sweep range · `icmr_gain_frac`. The extractor reads these from the item spec; nothing
here is hard-coded to a particular circuit.

## ngspice function quick-reference

| want | ngspice |
|---|---|
| magnitude dB | `db(h)` or `vdb(out)` |
| linear magnitude | `mag(h)` / `vm(out)` |
| phase, wrapping (deg) | `vp(out)` |
| phase, continuous (rad) | `cph(h)` → `*180/pi` for deg |
| supply branch current | `i(vdd)` / `vdd#branch` |
| DM / CM excitation (CMRR) | DM `AC 0.5 0`/`AC 0.5 180`; CM `AC 1 0`/`AC 1 0` |
| device region (sat check) | `@m1[vds]`, `@m1[vdsat]` after `.op` |
