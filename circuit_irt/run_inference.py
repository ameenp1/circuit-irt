"""Inference runner: roster of models x item bank -> scored response matrix.

For each model: load -> generate n samples/item (batched on vLLM) -> score every
completion through the grading harness -> append one JSONL record per
(model, item, sample). Append+flush makes it **resumable** — essential on spot
pods: re-running skips finished (model, item, sample) cells.

  python -m circuit_irt.run_inference --limit 5 --max-items 50 --n-samples 3
  python -m circuit_irt.run_inference --models configs/models.yaml --n-samples 5

Output: data/responses.jsonl (the matrix) + a variance summary printed at the end.
Push to HF with `circuit_irt.datastore` once backend: hf is configured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from circuit_irt.paths import DATA, CONFIGS
from circuit_irt.respondent import evaluate_completion
from circuit_irt.models import load_respondent


def _load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r["model"], r["item_id"], r["sample"]))
            except Exception:
                pass
    return done


def run(specs, items, n_samples, out_path, resume=True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path) if resume else set()
    with open(out_path, "a") as f:
        for spec in specs:
            mid = spec["id"]
            pending = [it for it in items
                       if any((mid, it["item_id"], s) not in done for s in range(n_samples))]
            if not pending:
                print(f"[{mid}] already complete — skipping")
                continue
            print(f"[{mid}] {len(pending)} items x {n_samples} samples "
                  f"(backend={spec.get('backend', 'auto')}"
                  f"{', reasoning' if spec.get('reasoning') else ''})")
            # A model that won't load (gated/awaiting access, 404, OOM) or errors
            # mid-generation must NOT kill the roster — log it and move on. It has
            # no records, so a later re-run picks it up automatically.
            try:
                model = load_respondent(spec)
                completions = model.complete(pending, n_samples)
            except Exception as e:
                print(f"[{mid}] SKIPPED — {type(e).__name__}: {str(e)[:200]}")
                continue
            for it, samples in zip(pending, completions):
                for s, comp in enumerate(samples):
                    if (mid, it["item_id"], s) in done:
                        continue
                    r = evaluate_completion(comp, it)
                    f.write(json.dumps(dict(
                        model=mid, item_id=it["item_id"], sample=s,
                        family=it["family_id"], tier=it["tier"],
                        label=r["label"], reason=r["reason"],
                        graded=r["graded"], all_pass=bool(r["all_pass"]))) + "\n")
                    f.flush()
            del model
            try:
                import gc, torch
                gc.collect(); torch.cuda.empty_cache()
            except Exception:
                pass
    return out_path


def assemble(out_path):
    import pandas as pd
    return pd.DataFrame(json.loads(l) for l in open(out_path))


def variance_summary(df) -> dict:
    """Collapse to pass_rate[model,item]; report ability spread + item variance —
    the sanity check that the matrix discriminates (not all-pass / all-fail)."""
    pr = df.groupby(["model", "item_id"])["all_pass"].mean()
    item_var = pr.groupby("item_id").var().mean()                # across-model variance
    ability = df.groupby("model")["all_pass"].mean().round(3)
    parse_fail = df["label"].eq("parse_failure").mean()
    return dict(
        records=len(df), models=int(df["model"].nunique()),
        items=int(df["item_id"].nunique()), n_samples=int(df["sample"].nunique()),
        overall_pass_rate=round(float(df["all_pass"].mean()), 3),
        parse_failure_rate=round(float(parse_fail), 3),
        model_ability=ability.to_dict(),
        mean_across_model_item_variance=round(float(item_var), 4),
        label_distribution=df["label"].value_counts().to_dict())


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(CONFIGS / "models.yaml"))
    ap.add_argument("--bank", default=str(DATA / "candidate_bank.json"))
    ap.add_argument("--out", default=str(DATA / "responses.jsonl"))
    ap.add_argument("--limit", type=int, help="use first N models")
    ap.add_argument("--max-items", type=int, help="use first N items")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    specs = yaml.safe_load(open(args.models))["models"]
    if args.limit:
        specs = specs[:args.limit]
    items = json.load(open(args.bank))["items"]
    if args.max_items:
        items = items[:args.max_items]

    run(specs, items, args.n_samples, args.out, resume=not args.no_resume)
    print("\n" + json.dumps(variance_summary(assemble(args.out)), indent=1))


if __name__ == "__main__":
    _main()
