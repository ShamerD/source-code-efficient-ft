#!/bin/bash
#SBATCH --job-name="93-f-gosum"
#SBATCH --time=1-06:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="93-%a.out"
#SBATCH --error="93-%a.err"
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
    --sub_task go \
    --use_defaults \
    --lr 3e-5 \
    --warmup 2500 \
    --batch_size 32 \
    --res_fn results/plbart_summarize_go_fflora.txt \
    --apply_lora_ff \
    --lora_ff_alpha 16 \
    --lora_ff_r $idx \
    --seed 1

