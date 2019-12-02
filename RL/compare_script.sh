#!/bin/bash

mc_flag=0
isplit_flag=1
num_repeat=5
problem='localization'
script="rl_isplit_${problem}.py"

d=`date | sed 's/ //g' | sed 's/\./_/g' | sed 's/,/_/g'`
for i in `seq 1 ${num_repeat}`
do
    # For output file name
    output_dir="out/${problem}/${d}/${i}"
    mkdir -p ${output_dir}
    # Run MC if flag > 0
    if [[ $mc_flag -gt 0 ]]
    then
        echo "[Info] Iter. ${i} | Monte Carlo..."
        python3 ${script} MC --outdir ${output_dir} > ${output_dir}/${i}_log_MC.txt
    fi
    # Run ISplit if flag > 0
    if [[ $isplit_flag -gt 0 ]]
    then
        echo "[Info] Iter. ${i} | Importance Splitting..."
        python3 ${script} --search ISplit --episodes 1000 --step 100 --outdir ${output_dir} > ${output_dir}/${i}_log_ISplit.txt
    fi
done
