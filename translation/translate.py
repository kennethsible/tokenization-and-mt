import torch
from sacrebleu.metrics import BLEU, CHRF
from torch import Tensor

from translation.decoder import beam_search
from translation.manager import Batch, Lemmatizer, Manager, Tokenizer


def mbr_decoding(paths: Tensor, probs: Tensor, manager: Manager, *, metric: str) -> str:
    tokenizer = Tokenizer(manager.src_lang, manager.tgt_lang, manager.sw_model)
    candidates = [
        [tokenizer.detokenize(manager.vocab.denumberize(path.tolist()))] for path in paths
    ]

    min_risk, best_translation = -1, ''
    for reference in candidates:
        # R(t)=\sum_{t'}\mathbb{P}(t'\mid x_1\cdots x_n)\cdot C(t,t')
        risk = 0
        for candidate, hyp_prob in zip(candidates, probs):
            # if reference == candidate:
            #     continue
            match metric:
                case 'bleu':
                    hyp_cost = 1 - BLEU().corpus_score(candidate, [reference]).score
                case 'chrf':
                    hyp_cost = 1 - CHRF().corpus_score(candidate, [reference]).score
                case _:
                    raise NotImplementedError(f"'{metric}' is not supported.")
            risk += hyp_prob * hyp_cost
        if risk < min_risk or min_risk == -1:
            min_risk = risk
            # t^*=\mathrm{argmin}_t\,R(t)
            best_translation = reference[0]

    return best_translation


def translate(string: str, manager: Manager, *, mbr_metric: str | None = None) -> str:
    model, vocab, device = manager.model, manager.vocab, manager.device
    beam_size, max_length = manager.beam_size, manager.max_length * 2
    tokenizer = Tokenizer(manager.src_lang, manager.tgt_lang, manager.sw_model)
    src_words = ['<BOS>'] + tokenizer.tokenize(string) + ['<EOS>']

    model.eval()
    with torch.no_grad():
        if manager.dict and manager.freq:
            lemmatizer = Lemmatizer(f'{manager.src_lang}_core_news_sm', manager.sw_model)
            lem_data = next(lemmatizer.lemmatize([src_words[1:-1]]))
            src_spans, tgt_spans = manager.append_defs(src_words, list(zip(*lem_data)))
            src_nums = torch.tensor(vocab.numberize(src_words), device=device)
            dict_data = list(zip([src_spans], [tgt_spans]))
            if manager.dpe_embed:
                src_encs = model.encode(src_nums.unsqueeze(0), dict_mask=None, dict_data=dict_data)
            else:
                mask_size = src_nums.unsqueeze(-2).size()
                dict_mask = Batch.dict_mask_from_data(dict_data, mask_size, device)
                src_encs = model.encode(src_nums.unsqueeze(0), dict_mask=dict_mask, dict_data=None)
        else:
            src_nums = torch.tensor(vocab.numberize(src_words), device=device)
            src_encs = model.encode(src_nums.unsqueeze(0))
        if manager.beam_size:
            paths, probs = beam_search(manager, src_encs, beam_size, max_length=max_length)
        else:
            paths, probs = beam_search(manager, src_encs, max_length=max_length)
        if mbr_metric is None:
            out_nums = paths[probs.argmax()]
        else:
            return mbr_decoding(paths, probs, manager, metric=mbr_metric)

    return tokenizer.detokenize(vocab.denumberize(out_nums.tolist()))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dict', metavar='FILE_PATH', help='bilingual dictionary')
    parser.add_argument('--freq', metavar='FILE_PATH', help='frequency statistics')
    parser.add_argument('--sw-vocab', metavar='FILE_PATH', required=True, help='subword vocab')
    parser.add_argument('--sw-model', metavar='FILE_PATH', required=True, help='subword model')
    parser.add_argument('--model', metavar='FILE_PATH', required=True, help='translation model')
    parser.add_argument('--mbr-metric', help='decoding metric (MBR)')
    parser.add_argument('--input', metavar='FILE_PATH', help='detokenized input')
    args, unknown = parser.parse_known_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_state = torch.load(args.model, map_location=device)

    config = model_state['config']
    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and len(unknown) > i:
            option, value = arg[2:].replace('-', '_'), unknown[i + 1]
            try:
                config[option] = (int if value.isdigit() else float)(value)
            except ValueError:
                config[option] = value

    manager = Manager(
        config,
        device,
        model_state['src_lang'],
        model_state['tgt_lang'],
        args.model,
        args.sw_vocab,
        args.sw_model,
        args.dict,
        args.freq,
    )
    manager.model.load_state_dict(model_state['state_dict'])

    if device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')

    with open(args.input) as data_f:
        for string in data_f.readlines():
            print(translate(string, manager, mbr_metric=args.mbr_metric))


if __name__ == '__main__':
    main()
