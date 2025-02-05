import argparse
import os
import re

import sentencepiece as spm
from subword_nmt.apply_bpe import BPE
from tqdm import tqdm

from translation.manager import Lemmatizer


def normalize(file_path: str, src_lang: str, tgt_lang: str) -> str:
    os.system(
        f'sacremoses -l {src_lang} -j 4 normalize < {file_path}.{src_lang} > {file_path}.norm.{src_lang}'
    )
    os.system(
        f'sacremoses -l {tgt_lang} -j 4 normalize < {file_path}.{tgt_lang} > {file_path}.norm.{tgt_lang}'
    )
    return file_path + '.norm'


def tokenize(file_path: str, src_lang: str, tgt_lang: str, escape_xml: bool = False) -> str:
    os.system(
        f'sacremoses -l {src_lang} -j 4 tokenize {"-x" if escape_xml else ""} < {file_path}.{src_lang} > {file_path}.tok.{src_lang}'
    )
    os.system(
        f'sacremoses -l {tgt_lang} -j 4 tokenize {"-x" if escape_xml else ""} < {file_path}.{tgt_lang} > {file_path}.tok.{tgt_lang}'
    )
    return file_path + '.tok'


def learn_bpe(file_path: str, data_dir: str, src_lang: str, tgt_lang: str, merge_ops: int):
    os.system(
        f'cat {file_path}.{src_lang} {file_path}.{tgt_lang} | subword-nmt learn-bpe -s {merge_ops} -o {data_dir}/{src_lang}-{tgt_lang}.model'
    )


def apply_bpe(
    file_path: str, data_dir: str, src_lang: str, tgt_lang: str, dropout: float = 0.0, seed: int = 0
) -> str:
    args = []
    if dropout > 0:
        args.append(f'--dropout {dropout}')
        if seed > 0:
            args.append(f'--seed {seed}')
    os.system(
        f'subword-nmt apply-bpe {" ".join(args)} -c {data_dir}/{src_lang}-{tgt_lang}.model < {file_path}.{src_lang} > {file_path}.bpe.{src_lang}'
    )
    os.system(
        f'subword-nmt apply-bpe {" ".join(args)} -c {data_dir}/{src_lang}-{tgt_lang}.model < {file_path}.{tgt_lang} > {file_path}.bpe.{tgt_lang}'
    )
    os.system(
        f'paste {file_path}.bpe.{src_lang} {file_path}.bpe.{tgt_lang} > {file_path}.bpe.{src_lang}-{tgt_lang}'
    )
    if 'train' in file_path.split('/'):
        os.system(
            f'cat {file_path}.bpe.{src_lang} {file_path}.bpe.{tgt_lang} | subword-nmt get-vocab > {data_dir}/{src_lang}-{tgt_lang}.vocab'
        )
    return file_path + '.bpe'


def learn_spm(file_path: str, src_lang: str, tgt_lang: str, vocab_size: int, model_type: str):
    os.system(
        f'cat {file_path}.{src_lang} {file_path}.{tgt_lang} > {file_path}.{src_lang}-{tgt_lang}'
    )
    spm.SentencePieceTrainer.train(
        input=f'{file_path}.{src_lang}-{tgt_lang}',
        model_prefix=f'{src_lang}-{tgt_lang}',
        vocab_size=vocab_size,
        model_type=model_type,
    )


def apply_spm(file_path: str, src_lang: str, tgt_lang: str) -> str:
    sp = spm.SentencePieceProcessor(model_file=f'{src_lang}-{tgt_lang}.model')
    with open(f'{file_path}.{src_lang}') as in_f, open(f'{file_path}.spm.{src_lang}', 'w') as out_f:
        out_f.writelines(
            [' '.join(words) + '\n' for words in sp.encode_as_pieces(in_f.readlines())]
        )
    with open(f'{file_path}.{tgt_lang}') as in_f, open(f'{file_path}.spm.{tgt_lang}', 'w') as out_f:
        out_f.writelines(
            [' '.join(words) + '\n' for words in sp.encode_as_pieces(in_f.readlines())]
        )
    os.system(
        f'paste {file_path}.spm.{src_lang} {file_path}.spm.{tgt_lang} > {file_path}.spm.{src_lang}-{tgt_lang}'
    )
    return file_path + '.spm'


def apply_initial_filter(file_path: str, src_lang: str, tgt_lang: str):
    lines = []
    with open(f'{file_path}.{src_lang}') as src_f, open(f'{file_path}.{tgt_lang}') as tgt_f:
        for src_line, tgt_line in tqdm(list(zip(src_f.readlines(), tgt_f.readlines()))):
            src_line, tgt_line = src_line.rstrip(), tgt_line.rstrip()
            if len(src_line) > 0 and len(tgt_line) > 0 and src_line != tgt_line:
                src_line = re.sub(r'\s+', ' ', src_line)
                tgt_line = re.sub(r'\s+', ' ', tgt_line)
                lines.append(f'{src_line}\t{tgt_line}')
    with open(f'{file_path}.{src_lang}', 'w') as src_f, open(
        f'{file_path}.{tgt_lang}', 'w'
    ) as tgt_f:
        for unique_lines in list(dict.fromkeys(lines)):
            src_line, tgt_line = unique_lines.split('\t')
            src_f.write(src_line + '\n')
            tgt_f.write(tgt_line + '\n')


def apply_final_filter(file_path: str, max_length: int, len_ratio: int):
    data = []
    with open(file_path) as data_f:
        for line in tqdm(data_f.readlines()):
            src_line, tgt_line = line.split('\t')
            src_words, tgt_words = src_line.split(), tgt_line.split()
            if (
                len(src_words) <= max_length
                and len(tgt_words) <= max_length
                and len(src_words) / len(tgt_words) <= len_ratio
                and len(tgt_words) / len(src_words) <= len_ratio
            ):
                data.append(src_line + '\t' + tgt_line)
    with open(file_path, 'w') as data_f:
        data_f.writelines(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pair', required=True, help='language pair')
    parser.add_argument('--data-dir', required=True, help='data directory')
    parser.add_argument('--max-length', type=int, required=True, help='maximum length')
    parser.add_argument('--len-ratio', type=float, required=True, help='length ratio')
    parser.add_argument('--lemmatize', action='store_true', help='lemmatize source')
    subparsers = parser.add_subparsers(dest='cmd', help='BPE or SentencePiece')
    bpe_parser = subparsers.add_parser('bpe')
    bpe_parser.add_argument('--merge-ops', required=True, help='merge operations')
    bpe_parser.add_argument('--dropout', type=float, help='subword dropout')
    bpe_parser.add_argument('--seed', type=int, help='random seed')
    sp_parser = subparsers.add_parser('spm')
    sp_parser.add_argument('--vocab-size', required=True, help='vocab size')
    sp_parser.add_argument('--model-type', required=True, help='model type')
    args = parser.parse_args()

    src_lang, tgt_lang = args.lang_pair.split('-')
    data_dir = args.data_dir
    os.system(f'mkdir -p {data_dir}')

    print('\n[1/10] Fetching Training Data...')
    train_path = f'{data_dir}/train'
    os.system(f'mkdir -p {train_path}')
    train_path += '/train'
    if not os.path.isfile(f'{train_path}.{src_lang}'):
        raise FileNotFoundError(f'{train_path}.{src_lang}')
    os.system(f'wc -l {train_path}.{src_lang}')
    if not os.path.isfile(f'{train_path}.{tgt_lang}'):
        raise FileNotFoundError(f'{train_path}.{tgt_lang}')
    os.system(f'wc -l {train_path}.{tgt_lang}')

    print('\n[2/10] Pre-Filtering Training Data...')
    apply_initial_filter(train_path, src_lang, tgt_lang)
    os.system(f'wc -l {train_path}.{src_lang}')
    os.system(f'wc -l {train_path}.{tgt_lang}')

    print('\n[3/10] Normalizing Training Data...')
    train_path = normalize(train_path, src_lang, tgt_lang)

    print('\n[4/10] Tokenizing Training Data...')
    train_path = tokenize(train_path, src_lang, tgt_lang, escape_xml=True)

    print('\n[5/10] Fetching Validation Set')
    val_path = f'{data_dir}/val'
    os.system(f'mkdir -p {val_path}')
    val_path += '/val'
    if not os.path.isfile(f'{val_path}.{src_lang}'):
        raise FileNotFoundError(f'{val_path}.{src_lang}')
    os.system(f'wc -l {val_path}.{src_lang}')
    if not os.path.isfile(f'{val_path}.{tgt_lang}'):
        raise FileNotFoundError(f'{val_path}.{tgt_lang}')
    os.system(f'wc -l {val_path}.{tgt_lang}')

    print('\n[6/10] Normalizing Validation Set...')
    val_path = normalize(val_path, src_lang, tgt_lang)

    print('\n[7/10] Tokenizing Validation Set...')
    val_path = tokenize(val_path, src_lang, tgt_lang, escape_xml=True)

    print('\n[8/10] Fetching Test Set...')
    test_path = f'{data_dir}/test'
    os.system(f'mkdir -p {test_path}')
    test_path += '/test'
    if not os.path.isfile(f'{test_path}.{src_lang}'):
        raise FileNotFoundError(f'{test_path}.{src_lang}')
    os.system(f'wc -l {test_path}.{src_lang}')

    if args.cmd == 'bpe':
        print('\n[9/10] Learning and Applying BPE...')
        learn_bpe(train_path, data_dir, src_lang, tgt_lang, args.merge_ops)
        train_path = apply_bpe(train_path, data_dir, src_lang, tgt_lang, args.dropout, args.seed)
        val_path = apply_bpe(val_path, data_dir, src_lang, tgt_lang)
        os.system(f'wc -l {data_dir}/{src_lang}-{tgt_lang}.vocab')
        with open(f'{data_dir}/{src_lang}-{tgt_lang}.model') as model_f:
            sw_model = BPE(model_f)
    else:
        print('\n[11/10] Learning and Applying SentencePiece...')
        learn_spm(train_path, src_lang, tgt_lang, args.vocab_size, args.model_type)
        apply_spm(train_path, src_lang, tgt_lang)
        apply_spm(val_path, src_lang, tgt_lang)
        os.system(f'mv {src_lang}-{tgt_lang}.model {src_lang}-{tgt_lang}.vocab {data_dir}')
        os.system(f'wc -l {data_dir}/{src_lang}-{tgt_lang}.vocab')
        sw_model = spm.SentencePieceProcessor(f'{src_lang}-{tgt_lang}.model')

    print('\n[10/10] Post-Filtering Training Data...')
    apply_final_filter(f'{train_path}.{src_lang}-{tgt_lang}', args.max_length, args.len_ratio)
    os.system(f'wc -l {train_path}.{src_lang}-{tgt_lang}')

    if args.lemmatize:
        print('\n[-/10] Lemmatizing Source Data...')
        lemmatizer = Lemmatizer(f'{src_lang}_core_news_sm', sw_model)
        for file_path in (train_path, val_path):
            src_words = []
            with open(f'{file_path}.{src_lang}-{tgt_lang}') as src_f:
                for line in src_f.readlines():
                    src_line = line.split('\t')[0]
                    src_words.append(src_line.split())
            with open(f'{file_path.split(".")[0]}.lem.{src_lang}', 'w') as lem_f:
                for words, spans in tqdm(lemmatizer.lemmatize(src_words), total=len(src_words)):
                    lem_f.write(f"{' '.join(words)}\t{' '.join(map(str, spans))}\n")

    print('\nDone.')


if __name__ == '__main__':
    main()
