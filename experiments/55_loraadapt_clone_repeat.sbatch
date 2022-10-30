#!/bin/bash
#SBATCH --job-name="55-fa-clone"
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="55-%a.out"
#SBATCH --error="55-%a.err"
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
    --task clone \
    --sub_task none \
    --use_defaults \
    --batch_size 8 \
    --src_len 350 \
    --trg_len 350 \
    --res_fn results/codet5_base_clone_fflora_adapter.txt \
    --apply_adapter \
    --adapter_type houlsby \
    --adapter_size 8 \
    --apply_lora_ff \
    --lora_ff_alpha 16 \
    --lora_ff_r 8 \
    --seed $idx

