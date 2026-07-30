# Rebuttal code package

Anonymized, self-contained snapshot of the code used for the NeurIPS rebuttal
experiments. It bundles the main co-teaching method together with the baseline
reproductions and the analysis scripts behind the rebuttal tables.

## Contents

```
.
├── pipeline.sh, config_example.sh, environment.yml   main co-teaching driver + config
├── create_*.py, train_gnn_with_pseudo_labels.py,     core method code
│   vllm_infer.py, node_selection.py, ...
├── common/                                           shared dataloader / GNN encoders / prompts
├── GNN/                                               standalone GNN training (Round 0 + baseline)
├── rebuttal_code/                                     our rebuttal scripts (see its own README)
│   ├── baselines/    pkd, gaugllm, gnn, llm_prompting, gaj_multiround
│   ├── data_prep/    WebKB + SBERT feature builders, self-training patch
│   ├── diagnosis/    validation-selection, R(t) mechanism, error-Jaccard, no-selection
│   └── infra/        kexec.py orchestration helper
└── baselines/                                         official third-party repos (code only)
    ├── PKD-main/     GAugLLM-main/     Locle-main/     GLEM-main/
```

## Not bundled (by design)

- **LLaMA-Factory** (the LLM fine-tuning framework the pipeline calls) is not
  vendored here. Install it from `environment.yml` (`llamafactory==0.9.4`) and
  point `LF_DIR` at it. The main repo keeps a vendored copy; it is omitted from
  this package to keep it small and because it is unmodified upstream code.
- **Model weights, datasets, checkpoints, logs, and caches** are excluded. Set
  `BASE_MODEL_PATH` to a Meta-Llama-3-8B-Instruct snapshot and place dataset
  `.pt` files under `datasets/`.
- **LLaGA and the GNN-as-Judge reproduction** were run from separate trees. The
  GNN-as-Judge codebase is the upstream this method is forked from (available
  upstream). LLaGA can be added on request.

## Baseline fidelity

See `rebuttal_code/README.md` for the important caveat: the hand
reimplementations of PKD and the classical GNN family did not match the original
papers and are kept for reference only. Reported baseline numbers use the
official repos under `baselines/` and the GNN-as-Judge reproduction.

## Paths

All absolute paths have been replaced with neutral placeholders (e.g.
`/home/anon/...`). Update them to your environment before running.
