import json
import os
import re
import warnings
from argparse import ArgumentParser, Namespace
from itertools import product

QUEUES = [
    'gpu@@nlp-a10',
    'gpu@@nlp-gpu',
    'gpu@@csecri',
    'gpu@@crc_gpu',
]


def comment_section(section: str) -> str:
    return '\n'.join(f'# {line}' for line in section.splitlines()) + '\n'


def generate_header(job_name: str, args: Namespace) -> str:
    string = '#!/bin/bash\n\n'
    string += f'touch {args.model}/{job_name}.log\n'
    string += f'fsync -d 30 {args.model}/{job_name}.log &\n'
    string += '\nset -eo pipefail\n'
    string += 'export SACREBLEU_FORMAT=text\n'
    return string


def generate_main(job_name: str, args: Namespace, params: list[tuple] | None = None) -> str:
    string = 'uv run -m translation.main  \\\n'
    string += f'  --lang-pair {args.lang_pair} \\\n'
    string += f'  --train-data {args.train_data} \\\n'
    string += f'  --val-data {args.val_data} \\\n'
    string += f'  --sw-vocab {args.sw_vocab} \\\n'
    string += f'  --sw-model {args.sw_model} \\\n'
    string += f'  --model {args.model}/{job_name}.pt \\\n'
    string += f'  --log {args.model}/{job_name}.log \\\n'
    if args.seed:
        string += f'  --seed {args.seed} \\\n'
    if params:
        for option, value in params:
            string += f'  --{option} {value} \\\n'
    return string


def generate_translate(job_name: str, test_data: str, args: Namespace) -> str:
    src_lang, _ = args.lang_pair.split('-')
    if re.match(r'wmt[0-9]{2}', test_data):
        _, test_data = test_data.split(':')
    test_set = test_data.split('/')[-1]
    string = 'uv run -m translation.translate  \\\n'
    string += f'  --sw-vocab {args.sw_vocab} \\\n'
    string += f'  --sw-model {args.sw_model} \\\n'
    if args.mbr_sampling:
        string += f'  --mbr-metric {args.mbr_metric} \\\n'
        string += f'  --mbr-sampling {args.mbr_sampling} \\\n'
    string += f'  --model {args.model}/{job_name}.pt \\\n'
    string += f'  --input {test_data}.{src_lang} \\\n'
    string += f'  > {args.model}/{job_name}.{test_set}.hyp \n'
    return string


def generate_sacrebleu(job_name: str, test_data: str, args: Namespace) -> str:
    _, tgt_lang = args.lang_pair.split('-')
    wmt_set = test_refs = ''
    if re.match(r'wmt[0-9]{2}', test_data):
        wmt_set, test_data = test_data.split(':')
    if '{' in test_data:
        test_data, test_refs = re.findall(r'([^{]+)(?:\{(\d+(?:,\d+)+)\})?', test_data)
    test_set = test_data.split('/')[-1]
    string = ''
    metrics = ['bleu', 'chrf', 'ter', '--chrf-word-order 2']
    if wmt_set:
        string += f'echo -e "\\n{test_data} (BLEU)" >> {args.model}/{job_name}.log \n'
        string += f'sacrebleu -t {wmt_set} -l {args.lang_pair} -w 4 \\\n'
        string += f'  -i {args.model}/{job_name}.{test_set}.hyp \\\n'
        string += f'  -m {" ".join(metric.strip('"') for metric in metrics)} \\\n'
        string += f'  >> {args.model}/{job_name}.log \n'
    else:
        string += f'echo -e "\\n{test_data} (BLEU)" >> {args.model}/{job_name}.log \n'
        if test_refs:
            string += f'sacrebleu {" ".join(f"{test_data}{i}.{tgt_lang}" for i in test_refs.split(","))} -w 4 \\\n'
        else:
            string += f'sacrebleu {test_data}.{tgt_lang} -w 4 \\\n'
        string += f'  -i {args.model}/{job_name}.{test_set}.hyp \\\n'
        string += f'  -m {" ".join(metric.strip('"') for metric in metrics)} \\\n'
        string += f'  >> {args.model}/{job_name}.log \n'
    return string


def generate_comet(job_name: str, test_data: str, args: Namespace) -> str:
    src_lang, tgt_lang = args.lang_pair.split('-')
    wmt_set = test_refs = ''
    if re.match(r'wmt[0-9]{2}', test_data):
        wmt_set, test_data = test_data.split(':')
    if '{' in test_data:
        test_data, test_refs = re.findall(r'([^{]+)(?:\{(\d+(?:,\d+)+)\})?', test_data)
    test_set = test_data.split('/')[-1]
    if not wmt_set and test_refs:
        test_set += test_refs.split(',')[0]
        warnings.warn(f"COMET does not support multiple references; using '{test_set}'")
    string = ''
    if wmt_set:
        string += f'echo -e "\\n{test_data} (COMET)" >> {args.model}/{job_name}.log \n'
        string += f'comet-score --quiet --only_system -d {wmt_set}:{args.lang_pair} \\\n'
        string += f'  -t {args.model}/{job_name}.{test_set}.hyp \\\n'
        string += f'  >> {args.model}/{job_name}.log \n'
    else:
        string += f'echo -e "\\n{test_data} (COMET)" >> {args.model}/{job_name}.log \n'
        string += 'comet-score --quiet --only_system \\\n'
        string += f'  -s {test_data}.{src_lang} \\\n'
        string += f'  -r {test_data}.{tgt_lang} \\\n'
        string += f'  -t {args.model}/{job_name}.{test_set}.hyp \\\n'
        string += f'  >> {args.model}/{job_name}.log \n'
    return string


def generate_bertscore(job_name: str, test_data: str, args: Namespace) -> str:
    _, tgt_lang = args.lang_pair.split('-')
    test_refs = ''
    if re.match(r'wmt[0-9]{2}', test_data):
        _, test_data = test_data.split(':')
    if '{' in test_data:
        test_data, test_refs = re.findall(r'([^{]+)(?:\{(\d+(?:,\d+)+)\})?', test_data)
    test_set = test_data.split('/')[-1]
    string = f'echo -e "\\n{test_data} (BERTScore)" >> {args.model}/{job_name}.log \n'
    string += f'bert-score --rescale_with_baseline --lang {tgt_lang} \\\n'
    if test_refs:
        for i in test_refs.split(','):
            string += f'  -r {test_data}{i}.{tgt_lang} \\\n'
    else:
        string += f'  -r {test_data}.{tgt_lang} \\\n'
    string += f'  -c {args.model}/{job_name}.{test_set}.hyp \\\n'
    string += f'  >> {args.model}/{job_name}.log \n'
    return string


def generate_job_script(job_name: str, args: Namespace, params: list[tuple] | None = None) -> str:
    string = generate_header(job_name, args)
    string += '\n' + comment_section(generate_main(job_name, args, params))
    for test_data in args.test_data:
        string += '\n' + generate_translate(job_name, test_data, args)
        string += '\n' + generate_sacrebleu(job_name, test_data, args)
        string += '\n' + generate_comet(job_name, test_data, args)
        string += '\n' + generate_bertscore(job_name, test_data, args)
    return string


def qf_submit(job_name: str, args: Namespace) -> str:
    string = 'qf submit --queue ' + ' --queue '.join(QUEUES)
    string += f' --name {job_name} --deferred --'
    if args.email:
        string += f' -M {args.email} -m abe'
    string += f' -l gpu_card=1 {args.model}/{job_name}.sh'
    return string


def set_defaults(args):
    if args.train_data is None:
        args.train_data = f'{args.data_dir}/train.norm.bpe.{args.lang_pair}'
        if not os.path.exists(args.train_data):
            args.train_data = f'{args.data_dir}/train.norm.spm.{args.lang_pair}'
    if args.val_data is None:
        args.val_data = f'{args.data_dir}/val.norm.bpe.{args.lang_pair}'
        if not os.path.exists(args.val_data):
            args.val_data = f'{args.data_dir}/val.norm.spm.{args.lang_pair}'
    if args.test_data is None:
        src_lang, _ = args.lang_pair.split('-')
        args.test_data = [f'{args.data_dir}/test.{src_lang}']
    if args.sw_vocab is None:
        args.sw_vocab = f'{args.data_dir}/vocab.tsv'
        if not os.path.exists(args.sw_vocab):
            args.sw_vocab = f'{args.data_dir}/{args.lang_pair}.vocab'
    if args.sw_model is None:
        args.sw_model = f'{args.data_dir}/codes.tsv'
        if not os.path.exists(args.sw_model):
            args.sw_model = f'{args.data_dir}/{args.lang_pair}.model'
    return args


def main():
    parser = ArgumentParser()
    parser.add_argument('--lang-pair', required=True, help='language pair')
    parser.add_argument('--data-dir', required=True, help='data directory')
    parser.add_argument('--train-data', metavar='FILE_PATH', help='training data')
    parser.add_argument('--val-data', metavar='FILE_PATH', help='validation data')
    parser.add_argument('--test-data', nargs='+', metavar='FILE_PATH', help='test data')
    parser.add_argument('--sw-vocab', metavar='FILE_PATH', help='subword vocab')
    parser.add_argument('--sw-model', metavar='FILE_PATH', help='subword model')
    parser.add_argument('--mbr-metric', metavar='METRIC', help='evaluation metric')
    parser.add_argument('--mbr-sampling', metavar='METHOD', help='sampling method')
    parser.add_argument('--model', required=True, help='pytorch model')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--array', metavar='FILE_PATH', help='parameter array')
    parser.add_argument('--start', metavar='INDEX', type=int, default=1, help='starting index')
    parser.add_argument('--email', help='email address')
    args = set_defaults(parser.parse_args())

    os.system(f'mkdir -p {args.model}')
    if args.array:
        param_array = []
        with open(args.array) as json_file:
            for option, values in json.load(json_file).items():
                param_array.append([(option, value) for value in values])
        for i, params in enumerate(product(*param_array), start=args.start):
            job_name = f'{args.model}_{str(i).rjust(3, "0")}'
            with open(f'{args.model}/{job_name}.sh', 'w') as job_file:
                job_file.write(generate_job_script(job_name, args, params))
            os.system(qf_submit(job_name, args))
    else:
        job_name = args.model
        with open(f'{args.model}/{job_name}.sh', 'w') as job_file:
            job_file.write(generate_job_script(job_name, args))
        os.system(qf_submit(job_name, args))
    os.system('qf check')


if __name__ == '__main__':
    main()
