#!/bin/bash
# LLM-GNN Co-Teaching Configuration
# Copy this file to config.sh and modify according to your setup

# ===== WORKSPACE CONFIGURATION =====
export WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/workspace}"
export LF_DIR="${LF_DIR:-$(pwd)/LLaMA-Factory}"
export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
export ENV_NAME="${ENV_NAME:-lgct}"
export CONDA_SH="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"

# ===== MODEL PATHS =====
# Set BASE_MODEL_PATH to a local snapshot directory or a HuggingFace model id.
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
export TEMPLATE="llama3"  # Options: llama3, mistral, llama2, qwen

# ===== GNN CONFIGURATION =====
export GNN_TYPE="GCN"           # Options: GCN, GAT, SAGE, SGConv
export GNN_HIDDEN_DIM=64
export GNN_LAYERS=2
export GNN_DROPOUT=0.5
export GNN_LR=1e-2              # GNN initial training LR
export GNN_EPOCHS=500           # GNN initial training epochs
export GNN_PATIENCE=100

# ===== TRAINING HYPERPARAMETERS =====
# SFT (Supervised Fine-Tuning) - initial warm-up
export LEARNING_RATE_SFT=5e-6
export BATCH_SIZE_SFT=4
export EPOCHS_SFT=10

# LoRA
export LORA_RANK=8
export LORA_ALPHA=16
export GRAD_ACCUM_STEPS=1

# ===== CO-TEACHING CROSS-SELECTION =====
# R(t) is adaptive: R(t) = agreed_rate (no hyperparameters needed)

# GNN re-training with LLM pseudo-labels
export GNN_TEACH_ALPHA=0.3      # Weight of pseudo-label loss (vs original label loss)
export GNN_TEACH_LR=1e-3        # GNN LR during co-teaching rounds
export GNN_TEACH_EPOCHS=50      # Epochs per round (fewer = less overfit to current pseudo-labels)

# LLM re-training with GNN pseudo-labels
export LLM_TEACH_LR=2e-5        # Lower LR to prevent catastrophic forgetting
export LLM_TEACH_EPOCHS=2       # Fewer epochs per round

# ===== HARDWARE CONFIGURATION =====
export NUM_GPUS=4
export MAIN_PROCESS_PORT_BASE=29500
export VISIBLE_DEVICES="0,1,2,3"
export DEVICE="cuda:0"

# ===== CO-TEACHING MINI-BATCH =====
# Each round: random sample BATCH_SIZE nodes from all unlabeled → GNN+LLM predict → cross-select
export BATCH_SIZE_POOL=1500        # Nodes sampled per round (like mini-batch in original Co-Teaching)
