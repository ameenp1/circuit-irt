# circuit-irt

IRT-based measurement of LLM analog-circuit design ability (Regeneron STS 2027).
Models are asked to design SPICE circuits to a spec; a functional grader runs
each design through `ngspice -b`, extracts measured metrics (gain, bandwidth,
phase margin, CMRR, …), scores them against the spec, and a Bayesian IRT model
calibrates item difficulty and model ability.

## Layout

```
circuit_irt/              library package
  netlist_dsl.py          minimal netlist DSL + ngspice batch runner
  families.py             the four circuit-family spec schemas + difficulty axes
  harness.py              simulate → extract_metrics → score → classify
  generators.py           spec-template + paraphrase generators
  reference.py            reference-solution generation + bank verification
  memorization_probes.py  verbatim vs perturbed canonical-circuit probes
  metadata.py             per-item difficulty-axis tagging
  respondent.py           prompt → model completion → netlist → score
  paths.py                canonical data/config locations
tests/                    test_spec_recipes, test_classifier
examples/                 rc_lowpass_demo
configs/                  models.yaml (respondent model roster)
data/                     candidate_bank.json, item_metadata.{parquet,csv}  (generated)
docs/                     spec_glossary, families, prior_work_notes
```

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .          # makes `circuit_irt` importable everywhere
```

Requires `ngspice` on PATH (`brew install ngspice`). See `requirements.txt` for
the Intel-mac-constrained pins (torch 2.2.2 / numpy<2 / transformers 4.49).

## Run

Modules and tests are runnable directly; each library module has a self-test:

```bash
.venv/bin/python -m circuit_irt.harness        # grader smoke test
.venv/bin/python -m circuit_irt.reference      # (re)build data/candidate_bank.json
.venv/bin/python -m circuit_irt.metadata       # (re)build data/item_metadata.*
.venv/bin/python tests/test_classifier.py      # failure-mode classifier unit tests
.venv/bin/python tests/test_spec_recipes.py    # metric-recipe validation vs closed form
RUN_MODELS=1 .venv/bin/python -m circuit_irt.respondent   # evaluate models.yaml
```

## Data

Generated artifacts (item bank, metadata, response matrices, frozen snapshots)
live in the local **`data/` folder**, which is gitignored — it's regenerable
from seeded code (`python -m circuit_irt.reference`). The backend is set in
`configs/data.yaml` (`backend: local` for now):

```bash
python -m circuit_irt.datastore status   # show backend + location
```

**Later**, flip `backend: hf` and set `hf.repo_id` to round-trip `data/` to a
HuggingFace dataset (`datastore push` / `pull`), with `hf.revision` pinning a
code version to an exact data version. HF's LFS handles the large response
matrices; reads/writes use `huggingface-cli login` or `HF_TOKEN`.

## Inference (GPU / RunPod)

Heavy inference (20-35 models, quantization, reasoning models) runs on a GPU pod,
not the Mac. The model roster + per-model metadata (scale, base/instruct &
code/general pairs for DIF, reasoning flags, quantization, sampling) is in
`configs/models.yaml`. Backends: `vllm` (GPU, batched), `hf` (transformers,
Mac/GPU), `mock` (synthetic, for runner tests).

On a RunPod (or any Linux+CUDA) pod:

```bash
bash scripts/setup_runpod.sh                 # ngspice + GPU deps + editable install
python -m circuit_irt.reference              # rebuild the item bank on the pod
python -m circuit_irt.run_inference --limit 5 --max-items 50 --n-samples 3   # smoke
python -m circuit_irt.run_inference --n-samples 5                            # full
```

The runner scores every completion through the grading harness and appends one
JSONL record per (model, item, sample) to `data/responses.jsonl` — **resumable**,
so a re-launched spot pod skips finished cells. `requirements-gpu.txt` /
`docker/Dockerfile.gpu` pin the CUDA stack. Reasoning traces (`<think>…</think>`)
are stripped before parsing.
