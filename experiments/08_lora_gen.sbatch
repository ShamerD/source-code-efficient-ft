#!/bin/bash
#SBATCH --job-name="08-l-gen"
#SBATCH --time=1-10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="08.out"
#SBATCH --error="08.err"

module purge
module load Python/Anaconda_v11.2021

source deactivate
source activate codelora

#Executable
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/
python ~/code-lora/CodeT5/sh/run.py \
    --model_tag codet5_base \
    --task concode \
    --sub_task none \
    --use_defaults \
    --batch_size 24 \
    --res_fn results/codet5_base_concode_lora.txt \
    --apply_lora \
    --lora_alpha 16 \
    --lora_r 8 \
    --seed 1

