#!/bin/bash
#SBATCH --job-name="34-f-pysum"
#SBATCH --time=1-06:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="34-%a.out"
#SBATCH --error="34-%a.err"
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
    --task summarize \
    --sub_task python \
    --use_defaults \
    --batch_size 32 \
    --res_fn results/codet5_base_summarize_python_fflora.txt \
    --apply_lora_ff \
    --lora_ff_alpha 16 \
    --lora_ff_r 8 \
    --seed $idx

