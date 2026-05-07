#!/bin/bash
# set -e

## -----------------------------
## Path Definitions
## -----------------------------
PROJECT_ROOT="/root/autodl-tmp"

IMAGENET_PATH="${PROJECT_ROOT}/datasets/imagenet/train"
CACHED_PATH="${PROJECT_ROOT}/datasets/imagenet/cached/kl-f16-coda"

VAE_PATH="${PROJECT_ROOT}/pretrained_models/coda/kl16.safetensors"
VAE_LORA_PATH="${PROJECT_ROOT}/pretrained_models/coda/vae_ema.pth"
VAE_QUANTIZER_PATH="${PROJECT_ROOT}/pretrained_models/coda/quantizer_ema.pth"

LOAD_PATH="${PROJECT_ROOT}/ckpts/kl-f16-coda/mar_large/masked_coda_qknorm_swiglu_rope_rmsnorm_woshift_00100/checkpoint-last.pth"
SAVE_PATH="${PROJECT_ROOT}/ckpts/kl-f16-coda/mar_large/masked_coda_qknorm_swiglu_rope_rmsnorm_woshift_00100"
LOG_PATH="${PROJECT_ROOT}/logs"

## -----------------------------
## Automated Logging
## -----------------------------
mkdir -p "${LOG_PATH}"
LOG_FILE="${LOG_PATH}/train_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "========================================"
echo " Job Started: $(date)"
echo " Log file: ${LOG_FILE}"
echo "========================================"

## -----------------------------
## Environment Setup
## -----------------------------
# export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda/envs"
# export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda/pkgs"
# source /opt/conda/etc/profile.d/conda.sh
# conda config --prepend envs_dirs "${CONDA_ENVS_PATH}"
# conda config --prepend pkgs_dirs "${CONDA_PKGS_DIRS}"
# conda activate aebm || conda activate "${CONDA_ENVS_PATH}/aebm"
# conda info --envs
# pip install tensorboard tqdm scipy einops timm torch-fidelity opencv-python pytorch-lightning omegaconf

echo "===== Environment Check ====="
export OMP_NUM_THREADS=1
which python
echo "CONDA_PREFIX=${CONDA_PREFIX}"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.version.cuda}')
print(f'GPUs:    {torch.cuda.device_count()}')
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f'Arch:    {major}.{minor}') 
"
echo "=========================================="

## -----------------------------
## Auto-Detect GPU Count
## -----------------------------
export NPROC_PER_NODE=$(nvidia-smi -L | grep -c "GPU")
echo " Node: $(hostname)" 
echo " Auto-detected GPUs: ${NPROC_PER_NODE}" 
nvidia-smi -L 
echo "========================================"

## -----------------------------
## Execution
## -----------------------------
cd "${PROJECT_ROOT}/aebm"
echo "Starting caching..."

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    main_cache.py \
    --img_size 256 \
    --vae_mode coda \
    --vae_path "${VAE_PATH}" \
    --vae_lora_path "${VAE_LORA_PATH}" \
    --vae_quantizer_path "${VAE_QUANTIZER_PATH}" \
    --vae_embed_dim 16 \
    --batch_size 96 \
    --num_workers 16 \
    --data_path "${IMAGENET_PATH}" \
    --cached_path "${CACHED_PATH}"

echo "========================================"
echo " Job Finished: $(date)"
echo "========================================"
