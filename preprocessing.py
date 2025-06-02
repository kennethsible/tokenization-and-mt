import argparse
import os
import re

from re import Match

import sentencepiece as spm
import unicodedata
from tqdm import tqdm

from utils.data.translation.vulgate_loader import VulgateDataset


NEWLINE_PATTERN: str = r"\\n.+$"


def normalize(file_path: str, src_lang: str, tgt_lang: str) -> None:
    for lang in (src_lang, tgt_lang):
        os.system(
            f'sacremoses -l {lang} -j 4 normalize < {file_path}.{lang} > {file_path}.norm.{lang}'
        )


def tokenize(file_path: str, src_lang: str, tgt_lang: str, escape_xml: bool = False) -> None:
    for lang in (src_lang, tgt_lang):
        os.system(
            f'sacremoses -l {lang} -j 4 tokenize {"-x" if escape_xml else ""} < {file_path}.{lang} > {file_path}.tok.{lang}'
        )


def learn_bpe(file_path: str, data_dir: str, src_lang: str, tgt_lang: str, merge_ops: int) -> None:
    os.system(
        f'cat {file_path}.{src_lang} {file_path}.{tgt_lang} | subword-nmt learn-bpe -s {merge_ops} -o {data_dir}/codes.tsv'
    )


def apply_bpe(
    file_path: str,
    data_dir: str,
    src_lang: str,
    tgt_lang: str,
    bpe_dropout: int | None = None,
    random_seed: int | None = None,
) -> None:
    dropout_str = ''
    if bpe_dropout:
        dropout_str += f' --dropout {bpe_dropout}'
        if random_seed:
            dropout_str += f' --seed {random_seed}'
    for lang in (src_lang, tgt_lang):
        os.system(
            f'subword-nmt apply-bpe{dropout_str} -c {data_dir}/codes.tsv < {file_path}.{lang} > {file_path}.bpe.{lang}'
        )


def get_vocab(file_path: str, data_dir: str, src_lang: str, tgt_lang: str):
    os.system(
        f'cat {file_path}.{src_lang} {file_path}.{tgt_lang} | subword-nmt get-vocab > {data_dir}/vocab.tsv'
    )


def apply_initial_filter(
    file_path: str,
    src_lang: str,
    tgt_lang: str,
    filter_names: list[str],
    preprocessor_names: list[str],
) -> None:
    with open(f'{file_path}.{src_lang}') as src_f, open(f'{file_path}.{tgt_lang}') as tgt_f:
        lines = []
        for src_line, tgt_line in tqdm(list(zip(src_f.readlines(), tgt_f.readlines()))):
            src_line, tgt_line = src_line.rstrip(), tgt_line.rstrip()
            for preprocessor_name in preprocessor_names:
                if preprocessor_name == "apparatus":
                    src_line, tgt_line = apply_apparatus_preprocessor(src_line, tgt_line)
                elif preprocessor_name == "newline":
                    src_line, tgt_line = apply_newline_preprocessor(src_line, tgt_line)

            if len(src_line) > 0 and len(tgt_line) > 0 and src_line != tgt_line:
                src_line = re.sub(
                    r'\s+', ' ', src_line
                )  # replace string of whitespaces with single space
                tgt_line = re.sub(r'\s+', ' ', tgt_line)
                lines.append(f'{src_line}\t{tgt_line}')
    print(len(lines))
    for filter_name in filter_names:
        if filter_name == "psalms":
            apply_psalm_filter(lines)
        if filter_name == "alphabet":
            apply_alphabet_filter(lines)
    print(len(lines))
    with open(f'{file_path}.{src_lang}', 'w') as src_f, open(
        f'{file_path}.{tgt_lang}', 'w'
    ) as tgt_f:
        for unique_lines in list(dict.fromkeys(lines)):
            src_line, tgt_line = unique_lines.split('\t')
            src_f.write(src_line + '\n')
            tgt_f.write(tgt_line + '\n')


def apply_psalm_filter(lines: list[str]):
    psalms: VulgateDataset = VulgateDataset()
    psalm_lines: list[str] = []

    for line in tqdm(lines, desc="Filtering Psalm Translations"):
        source, target = line.split("\t")
        if (source, target) in psalms:
            psalm_lines.append(line)
        elif (target, source) in psalms:
            psalm_lines.append(line)

    print(f"Psalms slated for removal: {len(psalm_lines)}")
    for line in psalm_lines:
        source, target = line.split("\t")
        print(f"Removing Pair:\n{source}\n{target}\n")
        lines.remove(line)


def is_latin(line: str):
    for char in line:
        if (
            char.isalpha()
            and unicodedata.category(char)[0] == 'L'
            and unicodedata.category(char) != 'Lm'
        ):
            if 'LATIN' not in unicodedata.name(char, '').split():
                return False
    return True


def apply_alphabet_filter(lines: list[str]):
    count = 0
    for line in tqdm(lines, desc="Filtering Non-Latin Script"):
        if is_latin(line) is False:
            # print(f'Removing Pair:{line}')
            lines.remove(line)
            count += 1
    print(f'Number of lines removed: {count}')


def apply_apparatus_preprocessor(src_line: str, tgt_line: str) -> tuple[str, str]:
    for character in ("<", ">", "[", "]"):
        src_line = src_line.replace(character, "")
        tgt_line = tgt_line.replace(character, "")

    return src_line, tgt_line


def apply_newline_preprocessor(src_line: str, tgt_line: str) -> tuple[str, str]:
    src_match: Match = re.search(NEWLINE_PATTERN, src_line)
    if src_match is not None:
        src_line = src_line.strip(src_match.group(0))

    tgt_match: Match = re.search(NEWLINE_PATTERN, tgt_line)
    if tgt_match is not None:
        tgt_line = tgt_line.strip(tgt_match.group(0))

    return src_line, tgt_line


def apply_final_filter(data_file: str, max_length: int, len_ratio: int) -> None:
    data = []
    with open(data_file) as data_f:
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

    with open(data_file, 'w') as data_f:
        data_f.writelines(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pair', required=True, help='language pair')
    parser.add_argument('--data-dir', required=True, help='data directory')
    parser.add_argument('--max-length', type=int, required=True, help='maximum length')
    parser.add_argument('--len-ratio', type=int, required=True, help='length ratio')
    parser.add_argument(
        "--filters",
        type=str,
        required=False,
        nargs="*",
        default=[],
        choices=["psalms", "alphabet"],
        help='specific filters',
    )
    parser.add_argument(
        "--preprocessors",
        type=str,
        required=False,
        nargs="*",
        default=[],
        choices=["apparatus", "newline"],
        help="specific preprocessors",
    )
    subparsers = parser.add_subparsers(dest='cmd', help='method of subword tokenization')
    bpe_parser = subparsers.add_parser('bpe')
    bpe_parser.add_argument('--merge-ops', required=True, help='merge operations')
    bpe_parser.add_argument('--bpe-dropout', type=float, help='subword dropout')
    bpe_parser.add_argument('--random-seed', type=int, help='random seed')
    sp_parser = subparsers.add_parser('spm')
    sp_parser.add_argument('--vocab-size', required=True, help='vocab size')
    sp_parser.add_argument('--model-type', required=True, help='model type')
    args = parser.parse_args()

    src_lang, tgt_lang = args.lang_pair.split('-')
    data_dir = args.data_dir
    os.system(f'mkdir -p {data_dir}')

    print('\n[1/12] Fetching Training Data...')
    train_path = f'{data_dir}/train'
    os.system(f'mkdir -p {train_path}')
    train_path += '/train'
    if not os.path.isfile(f'{train_path}.{src_lang}'):
        raise FileNotFoundError(f'{train_path}.{src_lang}')
    os.system(f'wc -l {train_path}.{src_lang}')
    os.system(f'wc -l {train_path}.{tgt_lang}')

    print('\n[1/12] Filtering Training Data (Initial)...')
    apply_initial_filter(train_path, src_lang, tgt_lang, args.filters, args.preprocessors)
    os.system(f'wc -l {train_path}.{src_lang}')
    os.system(f'wc -l {train_path}.{tgt_lang}')

    print('\n[2/12] Normalizing Training Data...')
    normalize(train_path, src_lang, tgt_lang)
    file_prefix = '.norm'

    print('\n[3/12] Tokenizing Training Data...')
    tokenize(train_path + file_prefix, src_lang, tgt_lang, escape_xml=True)
    file_prefix += '.tok'

    print('\n[4/12] Fetching Validation Set')
    val_path = f'{data_dir}/val'
    os.system(f'mkdir -p {val_path}')
    val_path += '/val'
    if not os.path.isfile(f'{val_path}.{src_lang}'):
        raise FileNotFoundError(f'{val_path}.{src_lang}')
    os.system(f'wc -l {val_path}.{src_lang}')
    os.system(f'wc -l {val_path}.{tgt_lang}')

    print('\n[5/12] Normalizing Validation Set...')
    normalize(val_path, src_lang, tgt_lang)
    file_prefix = '.norm'

    print('\n[6/12] Tokenizing Validation Set...')
    tokenize(val_path + file_prefix, src_lang, tgt_lang, escape_xml=True)
    file_prefix += '.tok'

    print('\n[7/12] Fetching Test Set...')
    test_path = f'{data_dir}/test'
    os.system(f'mkdir -p {test_path}')
    test_path += '/test'
    if not os.path.isfile(f'{test_path}.{src_lang}'):
        raise FileNotFoundError(f'{test_path}.{src_lang}')
    os.system(f'wc -l {test_path}.{src_lang}')
    os.system(f'wc -l {test_path}.{tgt_lang}')

    print('\n[8/12] Normalizing Test Set...')
    normalize(test_path, src_lang, tgt_lang)
    file_prefix = '.norm'

    print('\n[9/12] Tokenizing Test Set...')
    tokenize(test_path + file_prefix, src_lang, tgt_lang, escape_xml=True)
    file_prefix += '.tok'

    if args.cmd == 'bpe':
        print('\n[10/12] Learning and Applying BPE...')
        learn_bpe(train_path + file_prefix, data_dir, src_lang, tgt_lang, args.merge_ops)
        apply_bpe(
            train_path + file_prefix,
            data_dir,
            src_lang,
            tgt_lang,
            args.bpe_dropout,
            args.random_seed,
        )
        for file_path in (val_path, test_path):
            apply_bpe(file_path + file_prefix, data_dir, src_lang, tgt_lang)
        file_prefix += '.bpe'
        get_vocab(train_path + file_prefix, data_dir, src_lang, tgt_lang)
        os.system(f'wc -l {data_dir}/vocab.tsv')
    else:
        print('\n[10/12] Learning and Applying SentencePiece...')
        for file_path in (train_path, val_path, test_path):
            file_path += file_prefix
            os.system(
                f'cat {file_path}.{src_lang} {file_path}.{tgt_lang} > {file_path}.{src_lang}-{tgt_lang}'
            )
        spm.SentencePieceTrainer.train(
            input=f'{train_path}{file_prefix}.{src_lang}-{tgt_lang}',
            model_prefix=f'{src_lang}-{tgt_lang}',
            vocab_size=args.vocab_size,
            model_type=args.model_type,
        )
        sp = spm.SentencePieceProcessor(model_file=f'{src_lang}-{tgt_lang}.model')
        for file_path in (train_path, val_path, test_path):
            with open(f'{file_path}{file_prefix}.{src_lang}') as in_f, open(
                f'{file_path}{file_prefix}.spm.{src_lang}', 'w'
            ) as out_f:
                out_f.writelines(
                    [' '.join(words) + '\n' for words in sp.encode_as_pieces(in_f.readlines())]
                )
            with open(f'{file_path}{file_prefix}.{tgt_lang}') as in_f, open(
                f'{file_path}{file_prefix}.spm.{tgt_lang}', 'w'
            ) as out_f:
                out_f.writelines(
                    [' '.join(words) + '\n' for words in sp.encode_as_pieces(in_f.readlines())]
                )
        file_prefix += '.spm'
        os.system(f'mv {src_lang}-{tgt_lang}.model {src_lang}-{tgt_lang}.vocab {data_dir}')
        os.system(f'wc -l {data_dir}/{src_lang}-{tgt_lang}.vocab')

    print('\n[11/12] Filtering Training Data (Final)...')
    for file_path in (train_path, val_path, test_path):
        file_path += file_prefix
        os.system(
            f'paste {file_path}.{src_lang} {file_path}.{tgt_lang} > {file_path}.{src_lang}-{tgt_lang}'
        )
    apply_final_filter(
        f'{train_path}{file_prefix}.{src_lang}-{tgt_lang}', args.max_length, args.len_ratio
    )
    os.system(f'wc -l {train_path}{file_prefix}.{src_lang}-{tgt_lang}')

    print('\nDone.')


if __name__ == '__main__':
    main()
