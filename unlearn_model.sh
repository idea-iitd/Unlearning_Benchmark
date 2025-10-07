#!/bin/bash

# List of datasets
datasets=("cora")

# List of unlearning ratio
ratios=(0.1)

# List of unlearning methods
methods=("MEGU")

base_model=("GCN")

for method in "${methods[@]}"; do
  for dataset in "${datasets[@]}"; do
    for ratio in "${ratios[@]}"; do
      for base in "${base_model[@]}"; do
        echo "Running for dataset=${dataset}, unlearning_ratio=${ratio}, method=${method}, base model=${base_model}"
        python GULib-master/main.py \
          --dataset_name "$dataset" \
          --base_model "$base" \
          --unlearning_methods "$method" \
          --attack True \
          --num_epochs 100 \
          --batch_size 64 \
          --unlearn_ratio "$ratio" \
          --num_runs 5 \
          --cal_mem True 
      done
    done
  done
done
