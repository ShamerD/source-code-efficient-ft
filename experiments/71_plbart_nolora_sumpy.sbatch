#!/bin/bash
#SBATCH --job-name="71-n-sumpy"
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="71.out"
#SBATCH --error="71.err"

module purge
module load Python/Anaconda_v11.2021

source deactivate
source activate codelora

#Executable
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/
python ~/code-lora/CodeT5/sh/run.py \
    --model_tag plbart \
    --task summarize \
    --sub_task python \
    --use_defaults \
    --batch_size 32 \
    --lr 3e-5 \
    --warmup 2500 \
    --res_fn results/plbart_summarize_python.txt \
    --seed 1

