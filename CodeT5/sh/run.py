#!/usr/bin/env python
import os
import argparse
from pathlib import Path

WORK_DIR = Path(__file__).absolute().resolve().parent.parent

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default='codet5_base',
                        choices=['roberta', 'codebert', 'bart_base', 'codet5_small', 'codet5_base', 'plbart'])

    parser.add_argument("--task", type=str, default='summarize', choices=['summarize', 'concode', 'translate',
                                                                          'refine', 'defect', 'clone', 'multi_task'])
    parser.add_argument("--sub_task", type=str, default='python')

    parser.add_argument("--use_defaults", action="store_true", help="Initialize with default parameters")

    # Train params (override defaults)
    parser.add_argument("--batch_size", type=int, default=None, help='Training batch_size')
    parser.add_argument("--lr", type=float, default=None, help='Learning rate')
    parser.add_argument("--src_len", type=int, default=None, help='Max source sequence length')
    parser.add_argument("--trg_len", type=int, default=None, help='Max target sequence length')
    parser.add_argument("--patience", type=int, default=None, help='Validation epochs patience')
    parser.add_argument("--epoch", type=int, default=None, help='Number of training epochs')
    parser.add_argument("--warmup", type=int, default=None, help='Number of warmup iterations')

    # Misc
    parser.add_argument("--res_fn", type=str, default='results/codet5.txt', help='file to write text results')
    parser.add_argument("--model_dir", type=str, default='saved_models', help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='wandb', help='directory to save wandb summary')
    parser.add_argument("--data_num", type=int, default=-1, help='number of data instances to use, -1 for full data')
    parser.add_argument("--train_data_num", type=int, default=-1, help='number of train instances, -1 for full data')
    parser.add_argument("--gpu", type=int, default=0, help='index of the gpu to use in a cluster')
    parser.add_argument("--seed", type=int, default=0, help='fix seed')

    # LoRA
    parser.add_argument("--apply_lora", action="store_true",
                        help="Apply LoRA")
    parser.add_argument("--lora_alpha", type=int, default=0,
                        help="scaling coefficient in LoRA")
    parser.add_argument("--lora_r", type=int, default=0,
                        help="internal LoRA dimension")

    parser.add_argument("--apply_lora_ff", action="store_true",
                        help="Apply FF LoRA")
    parser.add_argument("--lora_ff_alpha", type=int, default=0,
                        help="scaling coefficient in FF LoRA")
    parser.add_argument("--lora_ff_r", type=int, default=0,
                        help="internal FF LoRA dimension")

    parser.add_argument("--apply_adapter", action="store_true",
                        help="Apply adapter")
    parser.add_argument("--adapter_type", type=str, choices=['houlsby', ''], default='',
                        help="Adapter type")
    parser.add_argument("--adapter_size", type=int, default=0,
                        help="Adapter layer hidden dim")

    parser.add_argument("--train_clone_head", action="store_true",
                        help="Train clone head")

    return parser


def prepare_files(args):
    for d in [args.model_dir, args.summary_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    Path(args.res_fn).parent.mkdir(parents=True, exist_ok=True)

def prepare_params(args):
    params = {}
    if args.use_defaults:
        params = get_params_by_task_model(args.task, args.sub_task, args.model_tag)

    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.lr is not None:
        params['learning_rate'] = args.lr
    if args.src_len is not None:
        params['src_len'] = args.src_len
    if args.trg_len is not None:
        params['trg_len'] = args.trg_len
    if args.patience is not None:
        params['patience'] = args.patience
    if args.epoch is not None:
        params['epoch'] = args.epoch
    if args.warmup is not None:
        params['warmup'] = args.warmup

    if args.apply_lora:
        assert args.lora_alpha > 0 and args.lora_r > 0
        params['lora_alpha'] = args.lora_alpha
        params['lora_r'] = args.lora_r

    if args.apply_lora_ff:
        assert args.lora_ff_alpha > 0 and args.lora_ff_r > 0
        params['lora_ff_alpha'] = args.lora_ff_alpha
        params['lora_ff_r'] = args.lora_ff_r

    if args.apply_adapter:
        assert args.adapter_size > 0
        params['adapter_type'] = args.adapter_type
        params['adapter_size'] = args.adapter_size

    assert all(p in params for p in [
        'batch_size',
        'learning_rate',
        'src_len',
        'trg_len',
        'patience',
        'epoch',
        'warmup'
    ])
    return params


def get_params_by_task_model(task, sub_task, model_tag):
    params = {}

    if task == 'multi_task':
        # Total train data num = 1149722 (for all five tasks)
        if 'codet5_small' in args.model_tag:
            params['batch_size'] = 60
            params['learning_rate'] = 5e-5
            params['max_steps'] = 600000
            params['save_steps'] = 20000
            params['log_steps'] = 100
        else:
            params['batch_size'] = 25
            params['learning_rate'] = 5e-5
            params['max_steps'] = 800000
            params['save_steps'] = 20000
            params['log_steps'] = 100

        if args.data_num != -1:
            params['max_steps'] = 1000
            params['save_steps'] = 200
            params['log_steps'] = 50

        params['src_len'] = -1
        params['trg_len'] = -1
        params['epoch'] = -1
        params['patience'] = -1
        params['warmup'] = 1000

        return params

    if task == 'translate':
        # java-cs: Read 10300 examples, avg src len: 13, avg trg len: 15, max src len: 136, max trg len: 118
        # [TOKENIZE] avg src len: 45, avg trg len: 56, max src len: 391, max trg len: 404
        params['src_len'] = 320
        params['trg_len'] = 256
        params['epoch'] = 100
        params['patience'] = 5
    elif task == 'summarize':
        # ruby: Read 24927 examples, avg src len: 66, avg trg len: 12, max src len: 501, max trg len: 146
        # [TOKENIZE] avg src len: 100, avg trg len: 13, max src len: 1250, max trg len: 161
        # Python: Read 251820 examples, avg src len: 100, avg trg len: 11, max src len: 512, max trg len: 222
        # [TOKENIZE] avg src len: 142, avg trg len: 12, max src len: 2016, max trg len: 245
        # Javascript: Read 58025 examples, avg src len: 114, avg trg len: 11, max src len: 512, max trg len: 165
        # [TOKENIZE] avg src len: 136, avg trg len: 12, max src len: 3016, max trg len: 177
        params['src_len'] = 256
        params['trg_len'] = 128
        params['epoch'] = 15
        params['patience'] = 2
    elif task == 'refine':
        # small: Read 46680 examples, avg src len: 31, avg trg len: 28, max src len: 50, max trg len: 50
        # [TOKENIZE] avg src len: 50, avg trg len: 45, max src len: 129, max trg len: 121
        # medium:  Read 52364 examples, avg src len: 74, avg trg len: 73, max src len: 100, max trg len: 100
        # [TOKENIZE] avg src len: 117, avg trg len: 114, max src len: 238, max trg len: 238
        if sub_task == 'small':
            params['src_len'] = 130
            params['trg_len'] = 120
        elif sub_task == 'medium':
            params['src_len'] = 240
            params['trg_len'] = 240
        else:
            raise NotImplementedError
        params['epoch'] = 50
        params['patience'] = 5
    elif task == 'concode':
        # Read 100000 examples, avg src len: 71, avg trg len: 26, max src len: 567, max trg len: 140
        # [TOKENIZE] avg src len: 213, avg trg len: 33, max src len: 2246, max trg len: 264
        params['src_len'] = 320
        params['trg_len'] = 150
        params['epoch'] = 30
        params['patience'] = 3
    elif task == 'defect':
        # Read 21854 examples, avg src len: 187, avg trg len: 1, max src len: 12195, max trg len: 1
        # [TOKENIZE] avg src len: 597, avg trg len: 1, max src len: 41447, max trg len: 1
        params['src_len'] = 512
        params['trg_len'] = 3
        params['epoch'] = 10
        params['patience'] = 2
    elif task == 'clone':
        # Read 901028 examples, avg src len: 120, avg trg len: 123, max src len: 5270, max trg len: 5270
        # [TOKENIZE] avg src len: 318, avg trg len: 323, max src len: 15111, max trg len: 15111
        params['src_len'] = 400
        params['trg_len'] = 400
        params['epoch'] = 1
        params['patience'] = 2
    else:
        raise NotImplementedError

    params['batch_size'] = 32
    if 'codet5_small' in model_tag:
        if task == 'summarize' or task == 'translate' or (task == 'refine' and sub_task == 'small'):
            params['batch_size'] = 64
        elif task == 'clone':
            params['batch_size'] = 25
    else:
        if task == 'translate':
            params['batch_size'] = 25
        elif task == 'summarize':
            params['batch_size'] = 48
        elif task == 'clone':
            if model_tag in ['codebert', 'roberta']:
                params['batch_size'] = 16
            else:
                params['batch_size'] = 10

    params['learning_rate'] = 5e-5
    if task == 'concode':
        params['learning_rate'] = 1e-4
    elif task == 'defect':
        params['learning_rate'] = 2e-5

    params['warmup'] = 1000

    return params


def get_sub_tasks(task):
    sub_tasks = []
    if task == 'summarize':
        sub_tasks = ['ruby', 'javascript', 'go', 'python', 'java', 'php']
    elif task == 'translate':
        sub_tasks = ['java-cs', 'cs-java']
    elif task == 'refine':
        sub_tasks = ['small', 'medium']
    elif task in ['concode', 'defect', 'clone', 'multi_task']:
        sub_tasks = ['none']
    return sub_tasks


def prepare_run(args, params):
    if args.data_num == -1:
        data_tag = 'all'
        epoch = params['epoch']
    else:
        data_tag = str(args.data_num)
        epoch = 1

    additional_info = ""
    if args.apply_lora:
        additional_info += f"L_{args.lora_alpha}_{args.lora_r}"
    if args.apply_lora_ff:
        additional_info += f"FL_{args.lora_ff_alpha}_{args.lora_ff_r}"
    if args.apply_adapter:
        additional_info += f"A_{args.adapter_type[0]}_{args.adapter_size}"

    if args.task == 'multi_task':
        full_model_tag = '_'.join([
            args.model_tag,
            data_tag,
            'lr' + str(int(params['learning_rate'] * 1e5)),
            's' + str(params['max_steps']),
            additional_info,
            'seed' + str(args.seed)
        ])
    else:
        full_model_tag = '_'.join([
            args.model_tag,
            data_tag,
            'lr' + str(int(params['learning_rate'] * 1e5)),
            'bs' + str(params['batch_size']),
            'src' + str(params['src_len']),
            'trg' + str(params['trg_len']),
            'pat' + str(params['patience']),
            'e' + str(epoch),
            additional_info,
            'seed' + str(args.seed)
        ])

    if args.sub_task == 'none':
        output_dir = Path(args.model_dir) / args.task / full_model_tag
    else:
        output_dir = Path(args.model_dir) / args.task / args.sub_task / full_model_tag

    cache_dir = output_dir / 'cache_data'
    res_dir = output_dir / 'prediction'
    log_fn = output_dir / 'train.log'

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    if args.model_tag == 'roberta':
        model_type = 'roberta'
        tokenizer = 'roberta-base'
        model_path = 'roberta-base'
    elif args.model_tag == 'codebert':
        model_type = 'roberta'
        tokenizer = 'roberta-base'
        model_path = 'microsoft/codebert-base'
    elif args.model_tag == 'bart-base':
        model_type = 'bart'
        tokenizer = 'facebook/bart-base'
        model_path = 'facebook/bart-base'
    elif args.model_tag == 'codet5_small':
        model_type = 'codet5'
        tokenizer = 'Salesforce/codet5-small'
        model_path = 'Salesforce/codet5-small'
    elif args.model_tag == 'codet5_base':
        model_type = 'codet5'
        tokenizer = 'Salesforce/codet5-base'
        model_path = 'Salesforce/codet5-base'
    elif args.model_tag == 'plbart':
        model_type = 'plbart'
        tokenizer = 'uclanlp/plbart-base'
        model_path = 'uclanlp/plbart-base'
    else:
        raise NotImplementedError

    multi_task_aug = ""
    if args.task == 'multi_task':
        run_fn = str(WORK_DIR / 'run_multi_gen.py')
        multi_task_aug = f"--max_steps {params['max_steps']} --save_steps {params['save_steps']} --log_steps {params['log_steps']}"
    elif args.task == 'clone':
        run_fn = str(WORK_DIR / 'run_clone.py')
    elif args.task == 'defect':
        run_fn = str(WORK_DIR / 'run_defect.py')
    else:
        run_fn = str(WORK_DIR / 'run_gen.py')

    lora_cmd = ""
    if args.apply_lora:
        lora_cmd += f" --apply_lora --lora_alpha {params['lora_alpha']} --lora_r {params['lora_r']}"
    if args.apply_lora_ff:
        lora_cmd += f" --apply_lora_ff --lora_ff_alpha {params['lora_ff_alpha']} --lora_ff_r {params['lora_ff_r']}"
    if args.apply_adapter:
        lora_cmd += f" --apply_adapter --adapter_type {params['adapter_type']} --adapter_size {params['adapter_size']}"
    if args.train_clone_head:
        lora_cmd += f" --train_clone_head"

    #res_fn = Path(args.res_dir) / (args.task + '_' + args.model_tag + '.txt')

    cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} python {run_fn}"
    cmd += f" --do_train --do_eval --do_eval_bleu --do_test {multi_task_aug}"
    cmd += f" --task {args.task} --sub_task {args.sub_task} --model_type {model_type} --data_num {args.data_num} --train_data_num {args.train_data_num}"
    cmd += f" --num_train_epochs {epoch} --warmup_steps {params['warmup']} --learning_rate {params['learning_rate']} --patience {params['patience']}"
    cmd += f" --tokenizer_name={tokenizer} --model_name_or_path={model_path} --data_dir {str(WORK_DIR / 'data')}"
    cmd += f" --cache_path {str(cache_dir)} --output_dir {str(output_dir)} --summary_dir {str(args.summary_dir)}"
    cmd += f" --save_last_checkpoints --always_save_model --res_dir {str(res_dir)} --res_fn {args.res_fn}"
    cmd += f" --train_batch_size {params['batch_size']} --eval_batch_size {params['batch_size']}"
    cmd += f" --max_source_length {params['src_len']} --max_target_length {params['trg_len']}"
    cmd += f" --seed {args.seed}"
    cmd += lora_cmd
    cmd += f" 2>&1 | tee {log_fn}"

    return cmd


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()

    assert args.sub_task in get_sub_tasks(args.task)

    prepare_files(args)
    params = prepare_params(args)

    cmd = prepare_run(args, params)

    print("RUNNING:")
    print(cmd)
    os.system(cmd)
