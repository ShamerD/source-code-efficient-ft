#!/bin/bash
#SBATCH --job-name="75-n-clone"
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="75-1.out"
#SBATCH --error="75-1.err"

module purge
module load Python/Anaconda_v11.2021

source deactivate
source activate codelora

#Executable
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/
python ~/code-lora/CodeT5/sh/run.py \
    --model_tag plbart \
    --task clone \
    --sub_task none \
    --use_defaults \
    --batch_size 12 \
    --lr 3e-5 \
    --warmup 2500 \
    --src_len 350 \
    --trg_len 350 \
    --res_fn results/plbart_clone.txt \
    --seed 1

