#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --job-name=CD-HIT
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=8 --mem=10000M
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=5-00:00:00
#SBATCH --output=CD-HIT_%j.out
#Skipping many options! see man sbatch
# From here on, we can start our program
nvidia-smi
pwd
hostname
# ~/miniconda3/envs/deeploc_torch/bin/cd-hit-est -i /home/sxr280/DeepRBPLoc/new_data/lncRNA_all_data_seq_includingmouse.fasta -o /home/sxr280/DeepRBPLoc/new_data/lncRNA_all_data_seq_includingmouse_filtered.fasta -c 0.90 -n 9 -d 0 -M 0 -T 8
# ~/miniconda3/envs/deeploc_torch/bin/cd-hit-est -i /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq.fasta -o /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_filtered.fasta -c 0.90 -n 9 -d 0 -M 0 -T 8
#for lncRNA
~/miniconda3/envs/deeploc_torch/bin/cd-hit-est -i /home/sxr280/DeepRBPLoc/new_data/lncRNA/lncRNA_all_data_seq_deduplicated_splitfrom_pooled.fasta -o /home/sxr280/DeepRBPLoc/new_data/lncRNA/lncRNA_all_data_seq_deduplicated_splitfrom_pooled_filtered.fasta -c 0.9 -n 9 -d 0 -M 0 -T 8

# ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/new_data/cross_RNA_prediction.py
# ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/DeepRBPLoc/benchmark.py


