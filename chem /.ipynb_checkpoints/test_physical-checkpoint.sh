lr_list=(0.0001)
batch_size_list=(16)
dropout_list=(0.0 0.1)
weight_decay=(1e-5 1e-10)
temp_list=(1)
portion_list=(0.0)

cl='results/20250624/finetune_1+2_mask0.5/cl/lr_0.001_decay_1e-10_bs_64_dropout_0.1_beta_1.0/best_model.pth'
vdss='results/20250624/finetune_1+2_mask0.5/vdss/lr_0.001_decay_1e-10_bs_64_dropout_0.1_beta_1.0/best_model.pth'
t1_2='results/20250624/finetune_1+2_mask0.5/t1_2/lr_0.001_decay_1e-10_bs_64_dropout_0.1_beta_1.0/best_model.pth'
filename='checkpoint/20250328/mask/best_model.pth'
output_model_file='results/20250626/final_finetune_mask0.5-1/'

mkdir -p "logs/20250428/final_bayes/"
mkdir -p $output_model_file

for decay in "${weight_decay[@]}"; do
  for lr in "${lr_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
      for dropout in "${dropout_list[@]}"; do
        for temp in "${temp_list[@]}"; do
          for portion in "${portion_list[@]}"; do
            # 定义日志文件名
            log_file="logs/20250428/final_bayes/lr${lr}_bs${batch_size}_drop${dropout}_${temp}.log"
            echo "Running experiment: lr=${lr}, batch_size=${batch_size}, dropout_ratio=${dropout}"
            
            # 运行训练脚本，并将输出重定向到日志文件
            # python chem/test_physical_together.py --temp $temp --decay $decay --save $output_model_file --lr $lr --batch_size $batch_size --dropout_ratio $dropout --filename $filename> $log_file 2>&1
            python chem/test_physical.py --portion $portion --temp $temp --decay $decay --save $output_model_file --lr $lr --batch_size $batch_size --dropout_ratio $dropout --input_vdss_model_file $vdss --input_cl_model_file $cl --input_t1_2_model_file $t1_2
            # python chem/test_physical_copy.py --decay $decay --save $output_model_file --lr $lr --batch_size $batch_size --dropout_ratio $dropout --input_vdss_model_file $vdss --input_cl_model_file $cl --input_t1_2_model_file $t1_2> $log_file 2>&1
            # python chem/test_physical.py --portion $portion --temp $temp --decay $decay --save $output_model_file --lr $lr --batch_size $batch_size --dropout_ratio $dropout     #从头
            # 打印完成信息
            echo "Experiment lr=${lr}, batch_size=${batch_size}, dropout=${dropout} completed!"
          done
        done
      done
    done
  done
done

echo "All experiments completed!"