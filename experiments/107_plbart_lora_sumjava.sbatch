#!/bin/bash
#SBATCH --job-name="107-l-javasum"
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="107-%a.out"
#SBATCH --error="107-%a.err"
#SBATCH --array=1,2,4,8,16

module purge
module load Python/Anaconda_v11.2021

source deactivate
source activate codelora

idx=$SLURM_ARRAY_TASK_ID

#Executable
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/
python ~/code-lora/CodeT5/sh/run.py \
    --model_tag plbart \
    --task summarize \
    --sub_task java \
    --use_defaults \
    --batch_size 32 \
    --lr 3e-5 \
    --warmup 2500 \
    --res_fn results/plbart_summarize_java_lora.txt \
    --apply_lora \
    --lora_alpha 16 \
    --lora_r $idx \
    --seed 1

