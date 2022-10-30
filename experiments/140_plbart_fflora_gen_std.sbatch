#!/bin/bash
#SBATCH --job-name="140-f-gen"
#SBATCH --time=1-12:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="140-%a.out"
#SBATCH --error="140-%a.err"
#SBATCH --array=2,3

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
    --task concode \
    --sub_task none \
    --use_defaults \
    --batch_size 32 \
    --lr 3e-5 \
    --warmup 2500 \
    --res_fn results/plbart_concode_fflora.txt \
    --apply_lora_ff \
    --lora_ff_alpha 16 \
    --lora_ff_r 128 \
    --seed $idx

