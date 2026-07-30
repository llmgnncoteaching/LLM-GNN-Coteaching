#!/bin/bash
D=$1; G=$2
cd /home/anon/graph_token_bench/upstream/LLaGA
rm -rf checkpoints/${D}/llaga-qwen7b-mpnet-nc
bash run_llaga_qwen7b.sh $D $G
bash run_llaga_qwen7b_eval.sh $D $G
echo "ALLDONE_${D}" >> /home/anon/graph-token-reasoning/outputs/LLAGAQWEN7B_${D}.txt
