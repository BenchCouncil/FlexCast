#!/bin/bash

# ── HF download (public repo, no token needed. China users: export HF_ENDPOINT=https://hf-mirror.com) ──
HF_USERNAME="zuyangsammie"
HF_REPO="FlexCast"
HF_MODEL_NAME="FlexCast"
CKPT_FILE="checkpoint.pth"
LOCAL_CKPT="./checkpoints/${HF_MODEL_NAME}/${CKPT_FILE}"
mkdir -p "$(dirname "$LOCAL_CKPT")"

if [ ! -f "$LOCAL_CKPT" ]; then
    echo ">>> Downloading checkpoint from HuggingFace ..."
    HF_URL="https://huggingface.co/${HF_USERNAME}/${HF_REPO}/resolve/main/${HF_MODEL_NAME}/${CKPT_FILE}"
    # Try wget first (works behind TLS intercept proxies), fallback to hf_hub
    if command -v wget &>/dev/null; then
        wget -q --show-progress --no-check-certificate -O "$LOCAL_CKPT" "$HF_URL" 2>&1 || {
            echo ">>> wget failed, trying huggingface_hub..."
            python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='${HF_USERNAME}/${HF_REPO}', filename='${HF_MODEL_NAME}/${CKPT_FILE}', local_dir='./checkpoints')
"
        }
    else
        python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='${HF_USERNAME}/${HF_REPO}', filename='${HF_MODEL_NAME}/${CKPT_FILE}', local_dir='./checkpoints')
"
    fi

    if [ -s "$LOCAL_CKPT" ]; then
        echo ">>> Downloaded $(du -h "$LOCAL_CKPT" | cut -f1)"
    else
        echo ">>> ERROR: Download failed, checkpoint is empty or missing"
        exit 1
    fi
else
    echo ">>> Checkpoint already cached at $LOCAL_CKPT ($(du -h "$LOCAL_CKPT" | cut -f1))"
fi

# ── Inference ─────────────────────────────────────────────────────────────
python -u exp/exp_inference.py \
    --checkpoint "$LOCAL_CKPT" \
    --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather globaltemp \
    --batch_size 32 \
    --device cuda
