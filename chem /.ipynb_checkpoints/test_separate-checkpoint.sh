lr_list=(0.0001)
batch_size_list=(32)
dropout_list=(0.1)
weight_decay=(1e-5)
portion_list=(1.0)
seed_list=(42)

cl='results/20250826/finetune_separate/cl/lr_0.001_decay_1e-05_bz_32_dropout_0.1/best_model.pth'
vdss='results/20250826/finetune_separate/vdss/lr_0.001_decay_1e-05_bz_32_dropout_0.1/best_model.pth'
t1_2='results/20250826/finetune_separate/t1_2/lr_0.001_decay_1e-05_bz_32_dropout_0.1/best_model.pth'
homo='results/20250909/finetune_mask/homo/lr_0.001_decay_1e-05_bz_32_dropout_0.1/best_model.pth'
lumo='results/20250909/finetune_mask/lumo/lr_0.001_decay_1e-05_bz_32_dropout_0.1/best_model.pth'
gap='results/20250909/finetune_mask/gap/lr_0.001_decay_1e-05_bz_32_dropout_0.1/best_model.pth'
filename='checkpoint/20250328/mask/best_model.pth'
output_model_file='results/20250909/final_separate_qm9/'

for seed in "${seed_list[@]}"; do
  for decay in "${weight_decay[@]}"; do
    for lr in "${lr_list[@]}"; do
      for batch_size in "${batch_size_list[@]}"; do
        for dropout in "${dropout_list[@]}"; do
          for portion in "${portion_list[@]}"; do
          
            # 定义日志文件名
            echo "Running experiment: lr=${lr}, batch_size=${batch_size}, dropout_ratio=${dropout}"
            
            # 运行训练脚本，并将输出重定向到日志文件
            # python chem/test_physical_together.py --temp $temp --decay $decay --save $output_model_file --lr $lr --batch_size $batch_size --dropout_ratio $dropout --filename $filename> $log_file 2>&1
            python chem/test_separate.py --runseed $seed --portion $portion --decay $decay --save $output_model_file --lr $lr --batch_size $batch_size --dropout_ratio $dropout --input_homo_model_file $homo --input_lumo_model_file $lumo --input_gap_model_file $gap
            # python chem/test_separate.py --runseed $seed --portion $portion --decay $decay --save $output_model_file --lr $lr --batch_size $batch_size --dropout_ratio $dropout --input_vdss_model_file $vdss --input_cl_model_file $cl --input_t1_2_model_file $t1_2
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