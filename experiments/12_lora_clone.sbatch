#!/bin/bash
#SBATCH --job-name="12-l-clone"
#SBATCH --time=0-02:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="12.out"
#SBATCH --error="12.err"

module purge
module load Python/Anaconda_v11.2021

source deactivate
source activate codelora

#Executable
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/
python ~/code-lora/CodeT5/sh/run.py \
    --model_tag codet5_base \
    --task clone \
    --sub_task none \
    --use_defaults \
    --batch_size 8 \
    --src_len 350 \
    --trg_len 350 \
    --res_fn results/codet5_base_clone_lora.txt \
    --apply_lora \
    --lora_alpha 16 \
    --lora_r 8 \
    --seed 1

