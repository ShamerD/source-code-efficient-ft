#!/bin/bash
#SBATCH --job-name="10-l-translate"
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="10.out"
#SBATCH --error="10.err"

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
    --res_fn results/codet5_base_translate_jc_lora.txt \
    --apply_lora \
    --lora_alpha 16 \
    --lora_r 8 \
    --seed 1

