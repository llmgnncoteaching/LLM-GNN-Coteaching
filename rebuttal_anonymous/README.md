# LLM-GNN Co-Teaching

Code accompanying the submission *LLM-GNN Co-Teaching: Breaking the Weak-Teacher Limit for Strong LLM-Based Graph Learning*.

This repository implements a bidirectional iterative co-teaching framework between a Graph Neural Network (GNN) and a Large Language Model (LLM) for few-shot node classification on text-attributed graphs.

## Overview

In each round both models predict on a shared unlabeled batch, rank their own predictions by an architecture-specific small-loss criterion (cross-entropy fit for the GNN, minimum token log probability for the LLM), and exchange their most confident pseudo-labels. Every two rounds we additionally run Round-based Pseudo-Label Preference Optimization (RPL-PO): nodes whose LLM prediction transitions from disagreeing-with-GNN to agreeing-with-GNN form a temporal preference pair, used to DPO-update the LLM.

## Repository Layout

```
.
├── pipeline.sh                       # End-to-end co-teaching driver
├── config_example.sh                 # Hyperparameter / path template (copy to config.sh)
├── environment.yml                   # Conda environment specification
├── create_sft.py                     # Build initial SFT data from few-shot split
├── create_co_teaching_data.py        # Cross-selection: build LLM/GNN pseudo-label exchange
├── create_preference_data.py         # Build cross-round DPO preference pairs
├── train_gnn_with_pseudo_labels.py   # GNN training with pseudo-label loss + EMA
├── vllm_infer.py                     # vLLM inference with token log-probabilities
├── select_influential_nodes.py       # Graph-based unlabeled-pool sampler
├── node_selection.py                 # Load LLM logprobs and select agreement nodes
├── evaluate_predictions.py           # Score LLM predictions against ground truth
├── common/                           # Shared utilities, GNN encoders, prompt templates
├── GNN/                              # Standalone GNN training / embedding scripts
└── LLaMA-Factory/                    # Vendored LLM fine-tuning framework
```

## Installation

```bash
# 1. Create the conda environment
conda env create -f environment.yml
conda activate lgct

# 2. Install LLaMA-Factory (vendored)
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..
```

Tested with Python 3.11, PyTorch 2.x, CUDA 12.x. Single-GPU runs require around 24 GB of VRAM. Multi-GPU configurations are supported via `accelerate`.

## Datasets

We evaluate on six text-attributed graphs: Cora, Citeseer, PubMed, WikiCS, ogbn-arxiv, and a curated subset of ogbn-products. Place pre-processed graph files at `datasets/<name>.pt`. Each `.pt` is a `torch_geometric.data.Data` object with attributes `x`, `y`, `edge_index`, `raw_texts`, `label_name`, `train_mask`, `val_mask`, `test_mask`. The dataset loader is in `common/dataloader.py`, and `create_sft.py` consumes these files directly to produce the few-shot split and initial SFT data.

## Quick Start

```bash
# 1. Configure
cp config_example.sh config.sh
# Edit config.sh: set BASE_MODEL_PATH, hardware settings, etc.

# 2. Run the full pipeline
#    Args: <dataset> <shots> <seed> <num_rounds> [BASE_MODEL_PATH]
./pipeline.sh cora 3 42 20
```

The driver builds the few-shot split, trains the initial GNN and LLM, then runs `<num_rounds>` of co-teaching with cross-round DPO on even rounds. Per-round metrics are written to `results/co_teaching/<dataset>_<shots>shot_seed<seed>/progress.csv`.

## Pipeline Stages

| Stage | Description | Script |
|---|---|---|
| 1 | Build few-shot split + initial SFT data | `create_sft.py` |
| 2 | Train initial GNN | `GNN/main.py` |
| 3 | SFT the initial LLM | LLaMA-Factory |
| 4 | Sample unlabeled batch | `select_influential_nodes.py` |
| 5a | LLM inference with logprobs | `vllm_infer.py` |
| 5b | Cross-selection of pseudo-labels | `create_co_teaching_data.py` |
| 5c | GNN training with pseudo-labels | `train_gnn_with_pseudo_labels.py` |
| 5d | LLM SFT with pseudo-labels | LLaMA-Factory |
| 5e | Evaluate | `evaluate_predictions.py` |
| 6 (even rounds only) | Build cross-round DPO pairs and run DPO | `create_preference_data.py` + LLaMA-Factory |

Stages 4–6 repeat per round.

## Reproducing the Main Table

For each `(dataset, shots)` setting:

```bash
./pipeline.sh <dataset> <shots> 42 20    # seed=42
./pipeline.sh <dataset> <shots> 7  20    # seed=7
./pipeline.sh <dataset> <shots> 13 20    # seed=13
```

Each run logs round-by-round LLM and GNN test accuracy. We report the round-best accuracy averaged across the three seeds. Hyperparameters are listed in `config_example.sh` and are shared across all six datasets at all three shot counts.

## Configuration

`config_example.sh` exposes the full hyperparameter set. Commonly tuned knobs:

- `GNN_TYPE`, `GNN_HIDDEN_DIM`, `GNN_LAYERS` — GNN architecture
- `LEARNING_RATE_SFT`, `EPOCHS_SFT`, `LORA_RANK`, `LORA_ALPHA` — initial LLM SFT
- `GNN_TEACH_ALPHA`, `GNN_TEACH_LR`, `GNN_TEACH_EPOCHS` — per-round GNN update
- `LLM_TEACH_LR`, `LLM_TEACH_EPOCHS` — per-round LLM update
- `BATCH_SIZE_POOL` — unlabeled sample size per round
- `NUM_GPUS`, `VISIBLE_DEVICES`, `DEVICE` — hardware

## Output Layout

```
results/co_teaching/<dataset>_<shots>shot_seed<seed>/
├── round0/
│   ├── gnn_round0.pt
│   └── sft/model/                 # initial LLM adapter
├── round1/
│   ├── gnn_round1.pt
│   ├── *_llm_preds.jsonl
│   ├── *_llm_teaches_gnn.json
│   ├── sft/model/
│   ├── test_predictions.jsonl
│   └── eval/round1/metrics.json
├── round2/
│   └── ...
└── progress.csv                   # per-round summary
```

## License

Released under the MIT License. See `LICENSE`.
