#!/usr/bin/env bash
# Download the 11 verified checkpoints (configs/lineages.yaml) to $CKPT_DIR.
# Run where the GPU lives (Colab via `colab exec`, or any box). Needs: pip install -U huggingface_hub
# Tulu base (meta-llama) is GATED: accept terms on HF + `huggingface-cli login` first.
set -euo pipefail
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
mkdir -p "$CKPT_DIR"

REPOS=(
  allenai/OLMo-2-1124-7B
  allenai/OLMo-2-1124-7B-SFT
  allenai/OLMo-2-1124-7B-DPO
  allenai/OLMo-2-1124-7B-Instruct
  meta-llama/Llama-3.1-8B
  allenai/Llama-3.1-Tulu-3-8B-SFT
  allenai/Llama-3.1-Tulu-3-8B-DPO
  allenai/Llama-3.1-Tulu-3-8B
  mistralai/Mistral-7B-v0.1
  HuggingFaceH4/mistral-7b-sft-beta
  HuggingFaceH4/zephyr-7b-beta
)

for repo in "${REPOS[@]}"; do
  echo "== $repo"
  huggingface-cli download "$repo" \
    --local-dir "$CKPT_DIR/${repo//\//__}" \
    --exclude "*.pth" "original/*"   # safetensors only; skip torch-pickle duplicates
done
echo "All checkpoints present under $CKPT_DIR"
