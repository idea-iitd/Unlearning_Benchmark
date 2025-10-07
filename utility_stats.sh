#!/bin/bash

# List of ratios
ratios=(0.1)

# List of datasets to run
datasets=("cora")

# List of unlearning methods
methods=("MEGU")

base_model=("GCN")

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        for ratio in "${ratios[@]}"; do
            for base in "${base_model[@]}"; do
                echo "Running for ratio=$ratio, dataset=$dataset, method=$method, base model=$base"
                python GULib-master/jaccard2.py \
                    --unlearn_task "node" \
                    --unlearn_ratio "$ratio" \
                    --dataset_name "$dataset" \
                    --unlearning_methods "$method" \
                    --num_runs 1 \
                    --base_model "$base" 
                echo -e "\n"
            done
        done
    done
done
