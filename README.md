# Parameter-Efficient Finetuning of Transformers for Source Code

# Description
Testing PE methods (LoRA, FF-LoRA, AT, FF-LoRA + AT)
with source code Transformers (CodeT5, PLBART)
on source code tasks (code summarization, code generation, code translation, code clone detection).

# Installation
```shell
./setup.sh
```

`LoRA/examples/NLU` contains source code of Transformers library

In order to add new method/model you need to do the following:
1. Modify Transformers' model code. See `LoRA/examples/NLU/src/transformers/models/plbart` for example.
2. Modify running scripts to support added method/model. See `CodeT5/configs.py`, `CodeT5/run_gen.py`, `CodeT5/sh/run.py`.

# Structure
- `LoRA/` contains source code for LoRA method
  - `LoRA/examples/NLU/` contains source code of Transformers library
- `CodeT5/` contains source codes for using models and running scripts
  - `CodeT5/sh/` contains running shell scripts with `CodeT5/sh/run.py` being main script
  - `CodeT5/run_*.py`, where `*` is in `[gen, clone, defect]`, is task-specific running script
- `experiments/` contains scripts which were used in our experiments

# Running experiments

If SLURM system is being used experiments can be run with
```shell
sbatch experiments/{id}_{model}_{method}_{task}_{info}.sbatch
```
where the pattern is approximately follows:
- `{id}` is the number of experiment;
- `{model}` is empty for CodeT5 and `plbart` for PLBART;
- `{method}` is the used Efficient Finetuning method `lora, fflora, adapt, comb (or loraadapt in earlier experiments)` or `clean` for full finetuning;
- `{task}` is the task: `sum(py|go|java)` for code summarization, `gen` for code generation, `translate_(cj|jc)` for code translation, `clone` for code clone detection;
- `{info}` is optional experiment-specific information

Otherwise, adapt script to shell syntax (see `experiments/README.md`)

Please, refer to `experiments/README.md` for details on how to reproduce each Figure or Table.