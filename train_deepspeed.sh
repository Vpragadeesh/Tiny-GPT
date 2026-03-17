#!/bin/bash
# train_deepspeed.sh - Launch training with DeepSpeed ZeRO-Infinity

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Tiny-GPT: DeepSpeed ZeRO-3 Training (CPU/NVMe Offloading)     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Training mode: passive (default) or aggressive
TRAIN_MODE=$(echo "${TRAIN_MODE:-passive}" | tr '[:upper:]' '[:lower:]')
export TRAIN_MODE
if [ "$TRAIN_MODE" != "passive" ] && [ "$TRAIN_MODE" != "aggressive" ]; then
    echo -e "${YELLOW}!${NC} Invalid TRAIN_MODE='$TRAIN_MODE'. Use 'passive' or 'aggressive'."
    exit 1
fi

# Check DeepSpeed installation
echo -e "${BLUE}[1/4]${NC} Checking dependencies..."
python -c "import deepspeed" 2>/dev/null && echo -e "      ${GREEN}✓${NC} DeepSpeed installed" || {
    echo -e "      Installing DeepSpeed..."
    pip install deepspeed -q
    echo -e "      ${GREEN}✓${NC} DeepSpeed installed"
}
echo -e "      ${GREEN}✓${NC} All dependencies ready"
echo

# Checkpoint handling (keep by default for auto-resume)
echo -e "${BLUE}[2/4]${NC} Checkpoint handling..."
mkdir -p checkpoints
if [ "${RESET_CHECKPOINTS:-0}" = "1" ]; then
    rm -f checkpoints/*.pt
    echo -e "      ${GREEN}✓${NC} Checkpoints cleared (RESET_CHECKPOINTS=1)"
else
    if ls checkpoints/*.pt >/dev/null 2>&1; then
        echo -e "      ${GREEN}✓${NC} Existing checkpoints found (auto-resume enabled)"
    else
        echo -e "      ${GREEN}✓${NC} No existing checkpoints (fresh run)"
    fi
fi
echo

# Verify dataset
echo -e "${BLUE}[3/4]${NC} Verifying dataset..."
if [ -f "data/train.bin" ] && [ -f "data/val.bin" ] && [ -f "data/test.bin" ]; then
    echo -e "      ${GREEN}✓${NC} Dataset ready"
else
    echo -e "      ${YELLOW}!${NC} Dataset not found. Run: python prepare_data.py"
    exit 1
fi
echo

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
fi
export NUM_GPUS

# Build active DeepSpeed config based on TRAIN_MODE
python - <<'PY'
import json
import os
import multiprocessing

mode = os.environ.get("TRAIN_MODE", "passive").lower()
num_gpus = int(os.environ.get("NUM_GPUS", "1"))

with open("ds_config.json") as f:
    cfg = json.load(f)

zero = cfg.setdefault("zero_optimization", {})
off_opt = zero.setdefault("offload_optimizer", {"device": "cpu"})
act_ckpt = cfg.setdefault("activation_checkpointing", {})

if mode == "aggressive":
    # Higher-throughput profile: larger batches and no CPU checkpointing.
    cfg["train_micro_batch_size_per_gpu"] = 2
    cfg["gradient_accumulation_steps"] = 8
    cfg["train_batch_size"] = cfg["train_micro_batch_size_per_gpu"] * cfg["gradient_accumulation_steps"] * max(1, num_gpus)
    off_opt["pin_memory"] = True
    zero["reduce_bucket_size"] = 2e6
    act_ckpt["cpu_checkpointing"] = False
else:
    # Low-resource profile (current stable baseline).
    cfg["train_micro_batch_size_per_gpu"] = 1
    cfg["gradient_accumulation_steps"] = 4
    cfg["train_batch_size"] = cfg["train_micro_batch_size_per_gpu"] * cfg["gradient_accumulation_steps"] * max(1, num_gpus)
    off_opt["pin_memory"] = False
    zero["reduce_bucket_size"] = 1e6
    act_ckpt["cpu_checkpointing"] = True

with open("ds_config.active.json", "w") as f:
    json.dump(cfg, f, indent=2)
PY

# Show configuration
echo -e "${BLUE}[4/4]${NC} Launching DeepSpeed training..."
python -c "
import json
with open('ds_config.active.json') as f:
    cfg = json.load(f)
print('  DeepSpeed Configuration:')
print(f\"    • Mode: ${TRAIN_MODE}\")
print(f\"    • ZeRO Stage: {cfg['zero_optimization']['stage']}\")
print(f\"    • Optimizer Offload: {cfg['zero_optimization']['offload_optimizer']['device']}\")
param_offload = cfg['zero_optimization'].get('offload_param', {}).get('device', 'none')
print(f\"    • Parameter Offload: {param_offload}\")
print(f\"    • Mixed Precision: {'bfloat16' if cfg.get('bf16', {}).get('enabled') else 'float32'}\")
print(f\"    • Micro Batch: {cfg['train_micro_batch_size_per_gpu']}\")
print(f\"    • Grad Accum: {cfg['gradient_accumulation_steps']}\")
print(f\"    • Batch Size: {cfg['train_batch_size']}\")
print()
"

# Launch training with DeepSpeed
echo -e "${YELLOW}Starting DeepSpeed training...${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Skip CUDA version mismatch check (system CUDA >= PyTorch CUDA is fine)
export DS_SKIP_CUDA_CHECK=1
if [ "$TRAIN_MODE" = "aggressive" ]; then
    CPU_THREADS=$(nproc)
    export MAX_JOBS=$CPU_THREADS
    export OMP_NUM_THREADS=$CPU_THREADS
    export MKL_NUM_THREADS=$CPU_THREADS
else
    export MAX_JOBS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
fi

# Main script reads this active config path.
export DS_CONFIG_PATH="ds_config.active.json"

# Launch with deepspeed
deepspeed --num_gpus $NUM_GPUS main_deepspeed.py

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}Training complete!${NC}"
echo
echo "Check results:"
echo "  • Checkpoints: ls -lh checkpoints/"
echo "  • Generate: python run.py"
echo "  • Best model: checkpoints/best.pt"
