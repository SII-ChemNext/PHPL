#!/bin/bash

# 定义超参数列表
lr_list=(0.001)
batch_size_list=(32)
dropout_list=(0.1)
weight_decay=(1e-5 1e-10)
beta_list=(1.0)

split=scaffold
dataset1="cl"
dataset2="vdss"
dataset3="t1_2"
input_model="checkpoint/20250727/mask_0.5/best_model.pth"
input_model_cl="checkpoint/20250326/cl/lr_0.001_decay_0_bs_256_dropout_0.0/best_epoch.pth"
input_model_vdss="checkpoint/20250326/vdss/lr_0.001_decay_0_bs_256_dropout_0.0/best_epoch.pth"
input_model_t1_2="checkpoint/20250326/t1_2/lr_0.001_decay_0_bs_256_dropout_0.0/best_epoch.pth"
model_save_dir="results/20250730/finetune_1+2_mask0.5_desc/"
# 创建日志目录
mkdir -p logs/0603/finetune1+2/

# 遍历所有超参数组合
for beta in "${beta_list[@]}"; do
  for lr in "${lr_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
      for dropout in "${dropout_list[@]}"; do
        for decay in "${weight_decay[@]}"; do
          # 定义日志文件名

          echo "Running experiment: lr=${lr}, batch_size=${batch_size}, dropout=${dropout}, beta${beta}"

          # 第一组数据集和保存路径
          # python chem/finetune_1+2_moe.py --input_model_file $input_model --beta $beta --decay $decay --split $split --gnn_type gin --save $model_save_dir --lr $lr --batch_size $batch_size --dropout_ratio $dropout > "$log_file1" 2>&1
          # python chem/finetune_1+2_moe.py --beta $beta --decay $decay --split $split --gnn_type gin --save $model_save_dir --lr $lr --batch_size $batch_size --dropout_ratio $dropout > "$log_file1" 2>&1
          # python chem/finetune_physical.py --decay $decay --input_model_file $input_model --split $split --gnn_type gin --dataset $dataset1 --save $model_save_dir1 --lr $lr --batch_size $batch_size --dropout_ratio $dropout > "$log_file1" 2>&1
          # python chem/finetune_1+2_together.py  --beta $beta --decay $decay --split $split --gnn_type gin  --save $model_save_dir --lr $lr --batch_size $batch_size --dropout_ratio $dropout 
          # python chem/finetune_1+2.py --input_model_cl $input_model_cl --input_model_vdss $input_model_vdss --input_model_cl $input_model_t1_2 --beta $beta --decay $decay --split $split --gnn_type gin --save $model_save_dir --lr $lr --batch_size $batch_size --dropout_ratio $dropout > "$log_file1" 2>&1
          python chem/finetune_1+2_desc.py --beta $beta --decay $decay --input_model_file $input_model --split $split --gnn_type gin --save $model_save_dir --lr $lr --batch_size $batch_size --dropout_ratio $dropout #加载mask用
          # python chem/finetune_1+2_desc.py --beta $beta --decay $decay --split $split --gnn_type gin --save $model_save_dir --lr $lr --batch_size $batch_size --dropout_ratio $dropout
          # python chem/finetune_1+2.py --beta $beta --decay $decay --input_model_file $input_model --split $split --gnn_type gin --save $model_save_dir --lr $lr --batch_size $batch_size --dropout_ratio $dropout #加载mask用
          # python chem/finetune_1+2.py --beta $beta --decay $decay --split $split --gnn_type gin --save $model_save_dir --lr $lr --batch_size $batch_size --dropout_ratio $dropout #从头用
          # torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 chem/finetune_1+2.py \
          #   --beta $beta --decay $decay --split $split --gnn_type gin --save $model_save_dir \
          #   --lr $lr --batch_size $batch_size --dropout_ratio $dropout > "$log_file1" 2>&1
          # 打印完成信息
          echo "Experiment lr=${lr}, batch_size=${batch_size}, dropout=${dropout} beta${beta} completed!"
        done
      done
    done
  done
done
echo "All experiments completed!"
