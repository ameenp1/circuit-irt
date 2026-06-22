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
import os
import subprocess
import sys
from pathlib import Path

# vLLM must spawn (not fork) its engine workers, or CUDA-already-initialized in
# the parent -> "Cannot re-initialize CUDA in forked subprocess". Set before any
# vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# hf-xet's concurrent writers fail on RunPod's network filesystem ("Background
# writer channel closed"); plain HTTPS downloads are reliable there. Override
# with HF_HUB_DISABLE_XET=0 on hosts where xet works.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

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
                        graded=r["graded"], all_pass=bool(r["all_pass"]),
                        netlist=r["netlist"],     # parsed netlist (or None)
                        completion=comp)) + "\n")  # raw output -> re-score offline, no re-run
                    f.flush()
            del model
            try:
                import gc, torch
                gc.collect(); torch.cuda.empty_cache()
            except Exception:
                pass
    return out_path


def rescore(in_path, bank_path, out_path):
    """Re-score stored completions against the current bank (new spec tightness,
    fixed labels, etc.) WITHOUT re-running any model. The whole point of saving
    `completion` in each record."""
    items = {it["item_id"]: it for it in json.load(open(bank_path))["items"]}
    n = 0
    with open(out_path, "w") as f:
        for line in open(in_path):
            rec = json.loads(line)
            it = items.get(rec["item_id"])
            if it is None or rec.get("completion") is None:
                continue
            r = evaluate_completion(rec["completion"], it)
            rec.update(label=r["label"], reason=r["reason"], graded=r["graded"],
                       all_pass=bool(r["all_pass"]), netlist=r["netlist"])
            f.write(json.dumps(rec) + "\n"); n += 1
    print(f"re-scored {n} completions -> {out_path}")
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
    ap.add_argument("--eager", action="store_true",
                    help="enforce_eager: skip vLLM CUDA-graph compile (fast smokes)")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score stored completions in --out against the bank (no model run)")
    ap.add_argument("--single-model",
                    help="(internal) run exactly one model in this process")
    ap.add_argument("--prune-cache", action="store_true",
                    help="delete each model's HF weights after it runs (bounds disk "
                         "to ~one model; needed on small-volume pods)")
    args = ap.parse_args()

    if args.rescore:
        out = rescore(args.out, args.bank, args.out + ".rescored")
        print("\n" + json.dumps(variance_summary(assemble(out)), indent=1))
        return

    specs = yaml.safe_load(open(args.models))["models"]
    if args.limit:
        specs = specs[:args.limit]
    if args.eager:
        for s in specs:
            s["enforce_eager"] = True
    items = json.load(open(args.bank))["items"]
    if args.max_items:
        items = items[:args.max_items]

    # --- worker mode: run exactly one model in THIS process, then exit ---------
    if args.single_model:
        spec = next(s for s in specs if s["id"] == args.single_model)
        run([spec], items, args.n_samples, args.out, resume=not args.no_resume)
        return

    # --- orchestrator: one SUBPROCESS per model so the OS reclaims all GPU VRAM
    # between models (a long-lived process leaks VRAM across vLLM loads). -------
    done = _load_done(Path(args.out)) if not args.no_resume else set()
    for spec in specs:
        mid = spec["id"]
        if all((mid, it["item_id"], s) in done
               for it in items for s in range(args.n_samples)):
            print(f"[{mid}] already complete — skipping")
            continue
        cmd = [sys.executable, "-m", "circuit_irt.run_inference",
               "--single-model", mid, "--models", args.models, "--bank", args.bank,
               "--out", args.out, "--n-samples", str(args.n_samples)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.max_items:
            cmd += ["--max-items", str(args.max_items)]
        if args.eager:
            cmd.append("--eager")
        print(f"\n=== launching {mid} in an isolated process ===", flush=True)
        subprocess.run(cmd)                       # fresh process -> VRAM freed on exit
        if args.prune_cache:
            _prune_model_cache(mid)               # free disk before the next download

    print("\n" + json.dumps(variance_summary(assemble(args.out)), indent=1))


def _prune_model_cache(model_id: str):
    """Delete a model's HF weights to keep disk bounded on small-volume pods."""
    import shutil
    hub = (os.environ.get("HF_HUB_CACHE")
           or (os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME")
               else os.path.expanduser("~/.cache/huggingface/hub")))
    d = os.path.join(hub, "models--" + model_id.replace("/", "--"))
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        print(f"    pruned cache: {d}", flush=True)


if __name__ == "__main__":
    _main()
