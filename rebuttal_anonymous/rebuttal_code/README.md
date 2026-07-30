# Rebuttal code

Consolidated code for the NeurIPS rebuttal experiments: baseline reproductions,
data preparation, and the per-table diagnosis/analysis scripts. Collected from
the run tree on our GPU cluster.

Track / table key:
- **item1** = validation-based round selection (AC gate 1)
- **T3** = higher-shot ogbn-arxiv (15/20/30/50) -> `tab:rebuttal_highshot`
- **T4** = full 47-class ogbn-products -> `tab:rebuttal_products47`
- **T5** = heterophilic WebKB (Cornell, Wisconsin) -> `tab:rebuttal_webkb`
- **T6** = selection ratio R(t) ablation -> `tab:rebuttal_rt`
- **T7** = bias accumulation / cross-teaching -> `tab:rebuttal_bias`, `tab:rebuttal_selftrain`

## Layout

```
rebuttal_code/
├── baselines/
│   ├── pkd/                 PKD (preference-driven KD) reproduction
│   │   ├── pkd_prep.py            teachers @ k-shot ONLY  (crippled, superseded)
│   │   ├── pkd_prep_faithful.py   teachers on LLM-expanded ~48% set (2-stage: annotate|train)
│   │   └── pkd_main_generic.py    node-selection + PPO teacher selector + distill + eval
│   ├── gaugllm/            GAugLLM (LLM-augmented graph contrastive learning)
│   │   ├── gaugllm_generate.py    LLM view generation -> SBERT-encode -> x_aug.pt (cache jsonl)
│   │   └── gaugllm_gcl.py         GCL train + few-shot linear probe
│   ├── gnn/
│   │   └── gnn_baselines.py       classical GCN/GAT/SAGE @ k-shot on standard test
│   ├── llm_prompting/
│   │   └── zs_webkb.py            Zero-shot + CoT (base Llama-3-8B, node text only)
│   └── gaj_multiround/
│       ├── pipeline_multiround.sh compute-matched multi-round GNN-as-Judge
│       └── config.sh
├── data_prep/
│   ├── build_webkb.py       Cornell/Wisconsin .pt from TAG CSVs + SBERT features (T5)
│   ├── build_sbert.py       generic SBERT feature builder
│   └── st_patch.py          self-training ablation patch for create_co_teaching_data.py (T7)
├── diagnosis/
│   ├── valsel_item1.py      offline validation-selection over per-round test preds (item1)
│   ├── rt_mechanism_T6.py   per-scheme selected-count vs pseudo-label quality (T6 mechanism)
│   ├── error_entrenchment_T7.py  round-to-round LLM error-set Jaccard (T7 confirmation bias)
│   └── noselect_stats_T7.py f10 no-selection arm stats (T7)
└── infra/
    └── kexec.py             REST exec into a remote Jupyter kernel (orchestration)
```

## How the pieces run

- **Baselines** run in a dedicated conda env on a GPU cluster, datasets in
  `LG-CoTeaching/datasets/*.pt`, model the Meta-Llama-3-8B-Instruct
  checkpoint. Each baseline was wrapped in a small sbatch (partition=normal)
  at run time.
- **Diagnosis scripts** are pure post-hoc analysis over the per-round
  `test_predictions.jsonl` / `progress.csv` in each run's
  `results/co_teaching/<ds>_<k>shot_seed42/` tree. No GPU needed.

## IMPORTANT caveat on baseline fidelity (read before reusing)

Hand reimplementation of PKD and the GNN family produced numbers **inconsistent
with the original papers** and should NOT be reported as-is:

- `pkd_prep_faithful.py` still gives cora 38.60 / wikics 38.20 (weak in-house LLM
  annotation ~48% vs the paper's 66% zero-shot, plus a distillation collapse).
- `gnn_baselines.py` gives arxiv GCN 31 vs the paper's verified 51 (SBERT vs OGB
  skip-gram feature mismatch).

The authoritative baselines for the paper come from the GNN-as-Judge reproduction
(GLEM/TAPE/LLM-GNN/LLaGA/GraphGPT/GAJ, verified reproducible). For faithful
PKD/GAugLLM/Locle/GLEM, use the **official repos** (bundled under
`../baselines/{PKD-main,GAugLLM-main,Locle-main,GLEM-main}`, also available
upstream). These reimplementations are kept for reference and for the parts
that did validate (Zero/CoT, R(t) and bias diagnosis, WebKB/products data prep).
