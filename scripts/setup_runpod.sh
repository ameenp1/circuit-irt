#!/usr/bin/env bash
# Provision a fresh RunPod (or any Linux+CUDA) pod for circuit-irt inference.
# Run from the repo root:  bash scripts/setup_runpod.sh
set -euo pipefail

echo ">> installing ngspice (system package)"
if command -v apt-get >/dev/null; then
  apt-get update -qq && apt-get install -y -qq ngspice git
fi

echo ">> python deps (GPU stack) + editable install"
pip install -U pip
pip install -r requirements-gpu.txt
pip install -e .

echo ">> HuggingFace auth"
# HF_TOKEN (read scope) is auto-read by huggingface_hub/transformers/vllm. We also
# cache it so `huggingface-cli` is logged in for the session. Gated models
# (meta-llama/*, google/gemma-2-*) additionally need access granted to this
# account on each model page — a token alone won't download them.
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 && echo "   logged in via HF_TOKEN"
else
  echo "   WARNING: HF_TOKEN not set — gated models (Llama, Gemma) will fail to download."
fi

echo ">> sanity checks"
ngspice --version | head -1
python - <<'PY'
import torch, circuit_irt
print("cuda available:", torch.cuda.is_available(),
      "| gpus:", torch.cuda.device_count())
import vllm; print("vllm:", vllm.__version__)
PY

cat <<'EOF'

ready. typical flow on the pod:
  # 1. get the item bank (regenerate, or pull from HF if configured)
  python -m circuit_irt.reference           # rebuilds data/candidate_bank.json
  # or: python -m circuit_irt.datastore pull   (needs backend: hf + HF_TOKEN)

  # 2. smoke test: 5 models x 50 items x 3 samples, then inspect variance
  python -m circuit_irt.run_inference --limit 5 --max-items 50 --n-samples 3

  # 3. full run (resumable; safe to re-launch on a spot pod)
  python -m circuit_irt.run_inference --n-samples 5

  # 4. push results to HF
  HF_TOKEN=... python -m circuit_irt.datastore push   # after backend: hf
EOF
