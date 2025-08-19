#!/bin/bash
#SBATCH --array=0-0%10
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#--nodelist=cc1gpu[001,002,003,004,005]
#--nodes=1
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=168:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Define an array of commands
experiments=(
    "python ../src/toy_data.py --alpha=1e-05 --batch_size=64 --criterion='L1' --delta=2.0 --deltaS=3 --epochs=1000 --embedding_level --experiments_directory='/cluster/tufts/hugheslab/eharve06/pooling/experiments/test' --lr=0.1 --model_name='alpha=1e-05_criterion=L1_lr=0.1_pooling=transformer_seed=1001_use_pos_embedding=False' --N_test=1000 --N_train=10000 --N_val=2500 --pooling='transformer' --save --seed=1001 --weight_decay=0.0"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate