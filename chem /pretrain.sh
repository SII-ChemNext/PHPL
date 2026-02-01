#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Hyper-parameter grid for pre-training
lr_list=(0.001)
batch_size_list=(32)
dropout_list=(0.1 0.0)
weight_decay=(1e-5 1e-10)
beta_list=(1.2)

split=scaffold
model_save_dir="$ROOT_DIR/results/20250913/pretrain_beta3/"
log_dir="$ROOT_DIR/logs/0603/pretrain/"

mkdir -p "$log_dir"
mkdir -p "$model_save_dir"

for beta in "${beta_list[@]}"; do
  for lr in "${lr_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
      for dropout in "${dropout_list[@]}"; do
        for decay in "${weight_decay[@]}"; do
          echo "Running experiment: lr=${lr}, batch_size=${batch_size}, dropout=${dropout}, beta=${beta}"

          python "$SCRIPT_DIR/pretrain_desc.py" \
            --beta "$beta" \
            --decay "$decay" \
            --split "$split" \
            --gnn_type gin \
            --save "$model_save_dir" \
            --lr "$lr" \
            --batch_size "$batch_size" \
            --dropout_ratio "$dropout"

          echo "Experiment lr=${lr}, batch_size=${batch_size}, dropout=${dropout}, beta=${beta} completed!"
        done
      done
    done
  done
done

echo "All experiments completed!"
