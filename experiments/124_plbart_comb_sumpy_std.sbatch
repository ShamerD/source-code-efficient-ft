#!/bin/bash
#SBATCH --job-name="124-fa-pysum"
#SBATCH --time=1-06:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="124-%a.out"
#SBATCH --error="124-%a.err"
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
    --task summarize \
    --sub_task python \
    --use_defaults \
    --batch_size 32 \
    --lr 3e-5 \
    --warmup 2500 \
    --res_fn results/plbart_summarize_python_fflora_adapter.txt \
    --apply_adapter \
    --adapter_type houlsby \
    --adapter_size 8 \
    --apply_lora_ff \
    --lora_ff_alpha 16 \
    --lora_ff_r 8 \
    --seed $idx

