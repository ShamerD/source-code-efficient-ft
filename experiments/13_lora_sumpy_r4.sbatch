#!/bin/bash
#SBATCH --job-name="13-l-pysum"
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="13-4.out"
#SBATCH --error="13-4.err"

module purge
module load Python/Anaconda_v11.2021

source deactivate
source activate codelora

#Executable
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/
python ~/code-lora/CodeT5/sh/run.py \
    --model_tag codet5_base \
    --task summarize \
    --sub_task python \
    --use_defaults \
    --batch_size 32 \
    --res_fn results/codet5_base_summarize_python_lora.txt \
    --apply_lora \
    --lora_alpha 16 \
    --lora_r 4 \
    --seed 1

