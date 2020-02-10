#!/bin/bash

script="main.py"
mc_flag=0
isplit_flag=1
num_repeat=5
problem='EKF'
simsteps=2000000
save_interval=100000
envparams='0.8'
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters - [optimizer, hidden layer initializer and activation]"
    exit
fi
opt=$1
init=$2
act=$3
agentparams="500000 1 -1 -1 ${opt} -1"
modelparams="${init} ${act} -1"
splitparams='100 10 -1 -1'

outdir_suffix="_EKF08_opt_${opt}_hidinit_${init}_hidact_${act}"
d=`date | sed 's/ //g' | sed 's/\./_/g' | sed 's/,/_/g'`
for i in `seq 1 ${num_repeat}`
do
    # For output file name
    output_dir="out/${problem}/${simsteps}/${d}_${outdir_suffix}/${i}"
    mkdir -p ${output_dir}

    # Run MC if flag > 0
    if [[ $mc_flag -gt 0 ]]
    then
        search='MC'
        echo "[Info] Iter. ${i} | Monte Carlo..."
    fi
    # Run ISplit if flag > 0
    if [[ $isplit_flag -gt 0 ]]
    then
        search='IS'
        echo "[Info] Iter. ${i} | Importance Splitting..."
    fi

    cmd="python3 ${script} --problem ${problem} --search ${search} --simsteps ${simsteps} --outdir ${output_dir} --interval ${save_interval} --envparams ${envparams} --agentparams ${agentparams} --splitparams ${splitparams} > ${output_dir}/${i}_log_${search}.txt"
    echo ${cmd}
    python3 ${script} --problem ${problem} --search ${search} --simsteps ${simsteps} --outdir ${output_dir} --interval ${save_interval} --envparams ${envparams} --agentparams ${agentparams} --splitparams ${splitparams} > ${output_dir}/${i}_log_${search}.txt

done
