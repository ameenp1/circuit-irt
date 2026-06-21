"""Week 4 inference smoke test (mock backend) + reasoning-trace parsing.

Validates the runner end-to-end without a GPU: 5 mock models (varying skill) x
50 stratified items x 3 samples -> scored response matrix -> sane variance
(ability spread, items discriminate). Also checks resumable checkpointing and
that reasoning-model (<think>…</think>) outputs parse. The real 4-5 model x 50
item GPU smoke runs identically on RunPod with the vllm backend.

Run:  .venv/bin/python tests/test_inference.py
"""
from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path

from circuit_irt.paths import DATA
from circuit_irt.respondent import parse_completion
from circuit_irt.run_inference import run, assemble, variance_summary


def _stratified(items, per_fam):
    by = defaultdict(list)
    for it in items:
        by[it["family_id"]].append(it)
    # cap the slow family (diff_pair: cmrr+icmr sweeps) for a quick smoke
    return (by["filters"][:per_fam] + by["cs_amp"][:per_fam]
            + by["two_stage_opamp"][:per_fam] + by["diff_pair"][:5])


def test_runner_and_variance():
    items = json.load(open(DATA / "candidate_bank.json"))["items"]
    subset = _stratified(items, 15)                       # ~50 items
    specs = [{"id": f"mock-skill-{int(s*100):02d}", "backend": "mock", "skill": s, "seed": i}
             for i, s in enumerate([0.2, 0.4, 0.6, 0.8, 0.95])]
    out = Path(tempfile.mkdtemp()) / "responses.jsonl"

    run(specs, subset, n_samples=3, out_path=out)
    df = assemble(out)
    summ = variance_summary(df)
    print("=== partial response matrix ===")
    print(json.dumps(summ, indent=1))

    assert summ["records"] == len(specs) * len(subset) * 3
    assert summ["models"] == 5 and summ["items"] == len(subset)
    # ability must increase with skill (the matrix discriminates models)
    ab = summ["model_ability"]
    order = [ab[f"mock-skill-{int(s*100):02d}"] for s in (0.2, 0.4, 0.6, 0.8, 0.95)]
    assert order == sorted(order), f"ability not monotonic in skill: {order}"
    assert order[-1] - order[0] > 0.3, "too little ability spread"
    # items must discriminate (non-trivial across-model variance)
    assert summ["mean_across_model_item_variance"] > 0.01, "items don't discriminate"

    # ---- resumable checkpointing: re-run is a no-op, no duplicate records ----
    n_before = len(df)
    run(specs, subset, n_samples=3, out_path=out)         # resume=True
    assert len(assemble(out)) == n_before, "resume duplicated work"
    print(f"\nresume check: {n_before} records unchanged on re-run.")


def test_reasoning_parse():
    nl = "R1 in out 1k\\nC1 out 0 159n"
    cases = {
        "R1 <think> then json":
            f"<think>\nNeed an RC low-pass; pick R=1k, C=159n so fc~1kHz.\n</think>\n\n"
            f'```json\n{{"netlist": "{nl}"}}\n```',
        "QwQ no tags, answer after reasoning":
            f'Let me reason... the cutoff sets RC. \n\nFinal answer:\n{{"netlist": "{nl}"}}',
        "think trace with a distractor brace inside":
            f"<think>maybe {{wrong: 1}} no</think>\n{{\"netlist\": \"{nl}\"}}",
    }
    for name, comp in cases.items():
        got = parse_completion(comp)
        print(f"  OK {name:<42} -> {'parsed' if got else 'None'}")
        assert got and "R1" in got, f"{name}: failed to parse"
    # truncated mid-reasoning (token cutoff, no answer) -> correctly no netlist
    assert parse_completion("<think> long reasoning that never reaches an answer") is None
    print("  OK truncated reasoning (no answer)         -> None (correct)")


if __name__ == "__main__":
    print("=== reasoning-trace parsing ===")
    test_reasoning_parse()
    print("\n=== runner + variance (mock backend) ===")
    test_runner_and_variance()
    print("\nOK: runner produces a discriminating matrix; resume works; "
          "reasoning traces parse.")
