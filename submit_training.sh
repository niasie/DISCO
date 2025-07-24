#!/bin/bash

#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --constraint=a100-40gb
#SBATCH --mem-per-cpu=2G
#SBATCH --time=120:00:00
#SBATCH --job-name=disco_training
#SBATCH --output=./experiments/slurm/disco_training_%A.out

source ./venv/bin/activate

python train.py \
--yaml_config operator_network_solver.yaml