#!/bin/bash
#SBATCH --job-name="09-n-translate"
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="09.out"
#SBATCH --error="09.err"

module purge
module load Python/Anaconda_v11.2021

source deactivate
source activate codelora

#Executable
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/
python ~/code-lora/CodeT5/sh/run.py \
    --model_tag codet5_base \
    --task translate \
    --sub_task java-cs \
    --use_defaults \
    --src_len 256 \
    --trg_len 200 \
    --res_fn results/codet5_base_translate-jc.txt \
    --seed 1

