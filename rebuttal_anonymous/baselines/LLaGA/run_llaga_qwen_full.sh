#!/bin/bash
D=$1; G=$2
cd /home/anon/graph_token_bench/upstream/LLaGA
rm -rf checkpoints/${D}/llaga-qwen3b-mpnet-nc
bash run_llaga_qwen.sh $D $G
bash run_llaga_qwen_eval.sh $D $G
echo "ALLDONE_${D}" >> /home/anon/graph-token-reasoning/outputs/LLAGAQWEN_${D}.txt
