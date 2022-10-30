#!/bin/bash
#SBATCH --job-name="lora-try-lora"
#SBATCH --time=1-00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output="03.out"
#SBATCH --error="03.err"

module purge
module load Python/Anaconda_v11.2021

source deactivate
source activate codelora

#Executable
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/
bash ~/code-lora/CodeT5/sh/exp_with_args.sh \
summarize python codet5_base \
0 -1 32 5 256 128 2 15 1000 \
saved_models wandb results/summarize_codet5_base_lora.txt \
1 16 8
