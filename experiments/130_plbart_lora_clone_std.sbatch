#!/bin/bash
#SBATCH --job-name="130-l-clone"
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="130-%a.out"
#SBATCH --error="130-%a.err"
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
    --task clone \
    --sub_task none \
    --use_defaults \
    --batch_size 12 \
    --lr 3e-5 \
    --warmup 2500 \
    --src_len 350 \
    --trg_len 350 \
    --res_fn results/plbart_clone_lora.txt \
    --apply_lora \
    --lora_alpha 16 \
    --lora_r 1 \
    --train_clone_head \
    --seed $idx

