#!/bin/bash
#SBATCH --job-name="131-f-clone"
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="131-%a.out"
#SBATCH --error="131-%a.err"
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
    --res_fn results/plbart_clone_fflora.txt \
    --apply_lora_ff \
    --lora_ff_alpha 16 \
    --lora_ff_r 16 \
    --train_clone_head \
    --seed $idx

