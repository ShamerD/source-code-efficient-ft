#!/bin/bash
#SBATCH --job-name="lora-try"
#SBATCH --time=1-00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output="02.out"
#SBATCH --error="02.err"
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda_v11.2021

source deactivate
source activate codelora

#Executable
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/
bash ~/code-lora/CodeT5/sh/exp_with_args.sh \
summarize python codet5_base \
0 -1 48 5 256 128 2 15 1000 \
saved_models wandb results/summarize_codet5_base.txt \
0 1 1
