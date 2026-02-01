#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

lr_list=(0.0001)
batch_size_list=(16)
dropout_list=(0.1)
weight_decay=(1e-5)
temp_list=(0.0)
portion_list=(0.0 0.2 0.4 0.6 0.8 1.0)

pretrain_checkpoint="$ROOT_DIR/results/20250913/pretrain_beta3/lr_0.001_decay_1e-05_bs_32_dropout_0.1_beta_0.0/best_model.pth"
output_model_dir="$ROOT_DIR/results/20251023/final_pre_auc_noisy/"
log_dir="$ROOT_DIR/logs/20250428/final_bayes/"

mkdir -p "$log_dir"
mkdir -p "$output_model_dir"

if [ ! -f "$pretrain_checkpoint" ]; then
  echo "Pretrained checkpoint not found at: $pretrain_checkpoint"
  echo "Please update pretrain_checkpoint to point to your generated best_model.pth."
  exit 1
fi


for decay in "${weight_decay[@]}"; do
  for lr in "${lr_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
      for dropout in "${dropout_list[@]}"; do
        for temp in "${temp_list[@]}"; do
          for portion in "${portion_list[@]}"; do
            echo "Running experiment: lr=${lr}, batch_size=${batch_size}, dropout_ratio=${dropout}, portion=${portion}, seed=${runseed}"

            python "$SCRIPT_DIR/finetune.py" \
              --runseed "$runseed" \
              --seed "$runseed" \
              --portion "$portion" \
              --temperature "$temp" \
              --decay "$decay" \
              --save "$output_model_dir" \
              --lr "$lr" \
              --batch_size "$batch_size" \
              --dropout_ratio "$dropout" \
              --filename "$pretrain_checkpoint"

            echo "Experiment lr=${lr}, batch_size=${batch_size}, dropout=${dropout} completed!"
          done
        done
      done
    done
  done
done


echo "All experiments completed!"
