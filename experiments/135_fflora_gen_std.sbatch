#!/bin/bash
#SBATCH --job-name="135-l-gen"
#SBATCH --time=1-12:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="135-%a.out"
#SBATCH --error="135-%a.err"
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
    --model_tag codet5_base \
    --task concode \
    --sub_task none \
    --use_defaults \
    --batch_size 24 \
    --res_fn results/codet5_base_concode_fflora.txt \
    --apply_lora_ff \
    --lora_ff_alpha 16 \
    --lora_ff_r 256 \
    --seed $idx

