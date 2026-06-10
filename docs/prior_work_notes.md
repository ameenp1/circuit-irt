# Prior-work notes: Masala-CHAI & SPICEPilot

Week 1 reading. Focus: netlist conventions + reusable task definitions, and how
their grading differs from this project's functional, spec-based grader.

Sources:
- Masala-CHAI — arXiv [2411.14299](https://arxiv.org/abs/2411.14299) (Nov 2024, rev. Mar 2025); [HTML v5](https://arxiv.org/html/2411.14299v5)
- SPICEPilot — arXiv [2410.20553](https://arxiv.org/abs/2410.20553) (Oct 2024); repo [ACADLab/SPICEPilot](https://github.com/ACADLab/SPICEPilot) (MIT)

---

## The one thing that matters most

**Both grade *structurally / syntactically*, not *functionally*.**

- **Masala-CHAI** verifies a netlist by (a) an LLM pass that fixes common errors
  (e.g. "removing floating nets") and (b) **Graph Edit Distance (GED)** vs. a
  reference topology (normalized 0–100% similarity; reports "100% similarity" vs
  AMSNet references). Pass@k means *the generated topology matches the intended
  one* — it does **not** simulate and check measured gain/bandwidth/etc.
- **SPICEPilot** scores Pass@k where "correct" ≈ *produces a valid netlist/PySpice
  script that runs*; a **human expert corrects trivial errors** before scoring.
  The paper explicitly concedes: *"while the framework generates accurate
  netlists… achieving functional efficiency and meeting key parameters such as
  gain requires further knowledge instillation."* No gain/BW/phase thresholds.

**Implication for us.** This is exactly the gap "Measuring, Not Scoring" targets.
Our grader runs `ngspice -b`, **extracts measured metrics** (gain, −3 dB BW, phase
margin, swing, quiescent power) and scores against **per-item spec targets** with a
smooth violation term. That is a functional grader, not a topology-match / "does it
compile" grader. → This is the honest novelty claim for the Week 18 framing: not
"first IRT-for-LLM" and not "first SPICE benchmark," but *spec-graded functional
measurement + IRT calibration*. State their structural grading plainly; claim only
the combination.

---

## Netlist conventions

### Masala-CHAI
- Plain **SPICE text** (their "industry-standard textual representation"); dataset
  entries are netlists + metadata (figure caption, component list, node list).
- **12 component classes** their detector recognizes — a good canonical element set
  to mirror in our DSL: *AC Source, BJT, Battery, Capacitor, DC Source, Diode,
  Ground, Inductor, MOSFET, Resistor, Current Source, Voltage Source.*
- Transistors named `M1, M2, …` (standard SPICE prefix-by-type). MOSFET models /
  subcircuits not detailed in the paper.
- Simulator **not pinned** ("SPICE and its variants" — ngspice/HSPICE). So their
  netlists are not guaranteed `ngspice -b`-clean without massaging.
- Dataset: **~7,500 schematics** extracted from **10 textbooks**, distributions
  reported over component count, node count, MOSFET count, lines-of-SPICE.

### SPICEPilot
- **Not raw netlists** — generates **PySpice Python objects** ("Python-based SPICE
  codes"). Graph view: nodes = components, edges = interconnections.
- → Reusing their reference solutions means **porting PySpice → flat `.cir` text**
  for our `ngspice -b` harness. PySpice also wraps `libngspice` (shared lib), which
  is the dependency we deliberately avoided. Treat their `.py` files as topology
  references, not drop-in decks.

### Our convention (decided, for the DSL — see `netlist_dsl.py`)
- Flat SPICE text, lowercase node names (`in`, `out`, `0` for ground), element
  helpers prefix-by-type (`V1`, `R1`, `C1`, `M1`…), analysis + `wrdata` in a
  `.control` block, run via `ngspice -b`. Adopt Masala-CHAI's 12-class element set
  as the element-helper roadmap.

---

## Reusable task definitions

### Masala-CHAI — 20-task generation benchmark + 7.5k dataset
Circuit types named (good menu for our **three families**, Week 2):
- Amplifiers: common-source, cascode, **differential**, 2-stage op-amp w/ Miller
  compensation, **telescopic cascode op-amp**, fully-differential amp w/ CMFB
- Biasing/refs: **current mirror**, **bandgap reference**
- Other: **filters**, LC oscillator, SRAM cell
- *Reuse status — checked [repo](https://github.com/jitendra-bhandari/Masala-CHAI):*
  **NOT safely reusable.** (a) **No license** — GitHub's license API returns 404 and
  there is no `LICENSE`/`COPYING` file in the tree, so it is **all-rights-reserved by
  default** despite the paper saying "open-source"; would need explicit author
  permission. (b) The repo `main` only ships ~488 **schematic JPGs**
  (`Dataset/data_1/0.jpg`…`487.jpg`), **not** the 7,500 SPICE netlists — the netlist
  corpus isn't in the repo (may be hosted off-repo; unconfirmed). Treat as a
  paper/topology *reference* only, not a data source.

### SPICEPilot — 24 tasks, MIT-licensed, in-repo
- Stored as **per-model PySpice `.py` files** under `Claude_tests/{easy,medium,hard}/`
  and `GPT_tests/{easy,medium,hard}/`; task list in `New_bench-mark.md`; system
  prompt in `Pilot_prompt.md`. No JSON/YAML schema; no separate runner in the tree.
- Task record = *circuit name + transistor count + short verbal description*
  (e.g. "SR Latch, 8T: two CMOS NOR gates"; "Operational Amplifier, 30T → Hard").
- **Difficulty = transistor count only**: Easy ≤10, Medium 11–25, Hard 26–45,
  Extreme >45.
- *Reusable:* MIT license lets us mine the **task list + prompt phrasing** and the
  reference topologies (after PySpice→`.cir` porting). Not reusable: their scoring.

---

## Decisions / takeaways for Week 2 scope-lock

1. **Difficulty:** do **not** copy transistor-count-only (SPICEPilot) or GED
   (Masala-CHAI). Keep our richer, IRT-calibrated axes (constraint-tightness,
   #simultaneous objectives, robustness/corner). Transistor count can be *one*
   logged metadata feature for the difficulty regression, not the definition.
2. **Three families** — strong candidates from the shared menu: (a) **op-amp /
   amplifier** (CS, cascode, diff-pair, 2-stage Miller), (b) **filters** (our RC
   demo already lives here), (c) **current-mirror / bias** (or LC oscillator) for a
   topology-vs-sizing contrast. Final call in Week 2.
3. **Element set:** mirror Masala-CHAI's 12 classes as the DSL helper roadmap.
4. **Reference harvesting:** SPICEPilot `.py` (**MIT, verified**) → port to `.cir`
   with attribution. Masala-CHAI is **off-limits as a data source** (no license =
   all-rights-reserved; netlists not even in the repo) — use only as a topology
   reference from the paper. Everything must be re-verified *functionally* through
   our harness, not trusted on import.
5. **Novelty framing (Week 18):** prior work = structural / Pass@k / human-corrected
   grading. Ours = functional spec-graded metrics + Bayesian IRT calibration.
