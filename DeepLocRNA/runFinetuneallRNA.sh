#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --job-name=DeepRBPLoc
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=8 --mem=80000M
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=3-00:00:00
#SBATCH --output=allRNA_finetune_DDP4GPUnonredundantDM3Loc9tflattenBCEhumanfiltermilncrelease45search0.0005_%j.out
#SBATCH --array=1  # Specify the range of job array tasks (corresponding to 10 layers)






#Skipping many options! see man sbatch
# From here on, we can start our program
nvidia-smi
pwd
hostname
#######SBATCH --output=allRNA_finetune_DDP4GPUredundant_%j.out
#####SBATCH --output=allRNA_finetune_DDP4GPUredundantMergeDM3Loc_%j.out
# Function to convert job array task ID to layer number
# layer_from_task_id() {
#   layers=(49 45 40 35 30 25 20 15 10 5)
#   echo "${layers[$1-1]}"
# }
# layer_from_task_id() {
#   layers=(40 35 30 25 20 15 10 5)
#   echo "${layers[$1-1]}"
# }
# layer_from_task_id() {
#   layers=(45 30 25 20 15 10 5)
#   echo "${layers[$1-1]}"
# }
layer_from_task_id() {
  layers=(20)
  echo "${layers[$1-1]}"
}

# Get the layer corresponding to the current job array task ID
layer=$(layer_from_task_id $SLURM_ARRAY_TASK_ID)


#nonredundant with dm3loc dataset
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer $layer --gpu_num $gpu_num --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --RNA_tag


#running mouse data
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer $layer --gpu_num $gpu_num --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_mouse_data_seq_mergedm3locall_deduplicated.fasta --load_data --RNA_tag


#runnning dm3loc  integration
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer $layer --gpu_num $gpu_num --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall.txt


#runnig different lr
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer $layer --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type BCE --jobnum $RANDOM --lr 0.0005
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer $layer --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type BCE --jobnum $RANDOM --lr 0.0001

# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer $layer --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type BCE --headnum 5 --fc_dim 150 --dim_attention 80 --jobnum $RANDOM --lr 0.0005 
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer $layer --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type BCE --headnum 8 --fc_dim 150 --dim_attention 80 --jobnum $RANDOM --lr 0.0005 
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer $layer --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type BCE --headnum 5 --fc_dim 200 --dim_attention 80 --jobnum $RANDOM --lr 0.0005 
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer $layer --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type BCE --headnum 5 --fc_dim 400 --dim_attention 80 --jobnum $RANDOM --lr 0.0005 
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer $layer --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type BCE --headnum 3 --fc_dim 500 --dim_attention 50 --jobnum $RANDOM --lr 0.0005 


#filtering the lnc and mi
#get new data
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 1 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type BCE --headnum 3 --fc_dim 500 --dim_attention 50 --jobnum $RANDOM --lr 0.0005 
#get tag data
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta --species human --batch_size 16 --flatten_tag --RNA_tag --gradient_clip --DDP --loss_type BCE --headnum 3 --fc_dim 500 --dim_attention 50 --jobnum $RANDOM --lr 0.0005 
#start to run the model
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta --load_data --species human --batch_size 16 --flatten_tag  --gradient_clip --DDP --loss_type fixed_weight --headnum 3 --fc_dim 500 --dim_attention 50 --jobnum $RANDOM --lr 0.0005 

# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta --load_data --species human --batch_size 16 --flatten_tag  --gradient_clip --DDP --loss_type BCE --headnum 3 --fc_dim 500 --dim_attention 50 --jobnum $RANDOM --lr 0.0005

#fra scratch
srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 45 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta --load_data --species human --batch_size 8 --flatten_tag  --gradient_clip --DDP --loss_type BCE --headnum 3 --fc_dim 500 --dim_attention 50 --jobnum $RANDOM --lr 0.0005


#running the mouse dataset
# srun ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 45 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_mouse_data_seq_deduplicated.fasta --load_data --species mouse --batch_size 8 --flatten_tag --gradient_clip --DDP --loss_type BCE --headnum 3 --fc_dim 500 --dim_attention 50 --jobnum $RANDOM --lr 0.0005
