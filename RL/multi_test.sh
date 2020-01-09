#!/bin/bash

optimizers='sgd adam'
hidden_initializer='glorot_uniform he_normal'
hidden_activations='relu leakyrelu sigmoid tanh'
optimizers='sgd'
hidden_initializer='glorot_uniform'
hidden_activations='relu'

for opt in ${optimizers}
do
    for init in ${hidden_initializer}
    do
        for act in ${hidden_activations}
        do
            echo "[Test] Optimizer: ${opt}, Hidden Initializer: ${init}, Hidden Activation: ${act}"
            ./compare_script.sh ${opt} ${init} ${act}
        done
    done
done
