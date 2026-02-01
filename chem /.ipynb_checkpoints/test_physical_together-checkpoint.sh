lr_list=(0.0001)
batch_size_list=(16)
dropout_list=(0.1)
weight_decay=(1e-5)
temp_list=(1.0)
portion_list=(0.8)
seed_list=(16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

# filename='results/20250825/finetune_1+2_betatest/lr_0.001_decay_1e-05_bs_32_dropout_0.0_beta_1.0/best_model.pth'
filename='results/20250723/finetune_1+2_0_desc/lr_0.001_decay_1e-05_bs_32_dropout_0.1_beta_1.0/best_model.pth'
output_model_file='results/20251017/final_pre_dataport/'

mkdir -p "logs/20250428/final_bayes/"
mkdir -p $output_model_file

for runseed in "${seed_list[@]}"; do
  for decay in "${weight_decay[@]}"; do
    for lr in "${lr_list[@]}"; do
      for batch_size in "${batch_size_list[@]}"; do
        for dropout in "${dropout_list[@]}"; do
          for temp in "${temp_list[@]}"; do
            for portion in "${portion_list[@]}"; do
            
              # 定义日志文件名
              echo "Running experiment: lr=${lr}, batch_size=${batch_size}, dropout_ratio=${dropout}"
              
              # 运行训练脚本，并将输出重定向到日志文件
              # python chem/test_physical_together_old.py --runseed $runseed --portion $portion --temperature $temp --decay $decay --save $output_model_file --lr $lr --batch_size $batch_size --dropout_ratio $dropout --filename $filename
              python chem/test_physical_together.py --runseed $runseed --seed $runseed --portion $portion --temperature $temp --decay $decay --save $output_model_file --lr $lr --batch_size $batch_size --dropout_ratio $dropout --filename $filename
              # python chem/test_physical_together.py --portion $portion --temperature $temp --decay $decay --save $output_model_file --lr $lr --batch_size $batch_size --dropout_ratio $dropout
              # python chem/test_physical_together.py --temperature $temp --decay $decay --save $output_model_file --lr $lr --batch_size $batch_size --dropout_ratio $dropout 
              # 打印完成信息
              echo "Experiment lr=${lr}, batch_size=${batch_size}, dropout=${dropout} completed!"
            done
          done
        done
      done
    done
  done
done


echo "All experiments completed!"