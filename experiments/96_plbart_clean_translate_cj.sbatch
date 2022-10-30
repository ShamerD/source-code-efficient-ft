#!/bin/bash
#SBATCH --job-name="96-n-translate"
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="96.out"
#SBATCH --error="96.err"

module purge
module load Python/Anaconda_v11.2021

source deactivate
source activate codelora

#Executable
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/
python ~/code-lora/CodeT5/sh/run.py \
    --model_tag plbart \
    --task translate \
    --sub_task cs-java \
    --use_defaults \
    --batch_size 32 \
    --lr 3e-5 \
    --warmup 2500 \
    --src_len 256 \
    --trg_len 200 \
    --res_fn results/plbart_translate_cj.txt \
    --seed 1

