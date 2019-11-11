#!/bin/bash

output_dir="out/10tests"
num_repeat=10

for i in `seq 1 ${num_repeat}`
do
    d=`date | sed 's/ //g' | sed 's/\./_/g' | sed 's/,/_/g'`
    echo "[Info] Iter. ${i} | Importance Splitting..."
    python3 rl_imp_splitting.py ISplit > ${output_dir}/${d}_log_ISplit.txt
    echo "[Info] Iter. ${i} | Monte Carlo..."
    python3 rl_imp_splitting.py MC > ${output_dir}/${d}_log_MC.txt
done
