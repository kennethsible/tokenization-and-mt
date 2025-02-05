import logging
import math
import random
import time
from datetime import timedelta

import tomllib
import torch
from tqdm import tqdm

from translation.manager import Batch, Manager

Criterion = torch.nn.CrossEntropyLoss
Optimizer = torch.optim.Optimizer
Scaler = torch.cuda.amp.GradScaler
Logger = logging.Logger


def train_epoch(
    data: list[Batch],
    manager: Manager,
    criterion: Criterion,
    optimizer: Optimizer | None = None,
    scaler: Scaler | None = None,
) -> float:
    total_loss, num_tokens = 0.0, 0
    for batch in tqdm(data):
        src_nums, src_mask = batch.src_nums, batch.src_mask
        tgt_nums, tgt_mask = batch.tgt_nums, batch.tgt_mask
        batch_length = batch.length()

        if manager.dpe_embed:
            dict_mask, dict_data = None, batch._dict_data
        else:
            dict_mask, dict_data = batch.dict_mask, None

        with torch.cuda.amp.autocast():
            logits = manager.model(src_nums, tgt_nums, src_mask, tgt_mask, dict_mask, dict_data)
            loss = criterion(torch.flatten(logits[:, :-1], 0, 1), torch.flatten(tgt_nums[:, 1:]))

        if optimizer and scaler:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                manager.model.parameters(),
                manager.clip_grad,
            )
            scaler.step(optimizer)
            scaler.update()

        total_loss += batch_length * loss.item()
        num_tokens += batch_length
        del logits, loss

    return total_loss / num_tokens


def train_model(train_data: list[Batch], val_data: list[Batch], manager: Manager, logger: Logger):
    model, vocab = manager.model, manager.vocab
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=vocab.PAD, label_smoothing=manager.label_smoothing
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=manager.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=manager.decay_factor, patience=manager.patience
    )
    scaler = torch.cuda.amp.GradScaler()

    epoch = patience = 0
    best_loss = torch.inf
    while epoch < manager.max_epochs:
        random.shuffle(train_data)

        model.train()
        start = time.perf_counter()
        train_loss = train_epoch(train_data, manager, criterion, optimizer, scaler)
        elapsed = timedelta(seconds=(time.perf_counter() - start))

        model.eval()
        with torch.no_grad():
            val_loss = train_epoch(val_data, manager, criterion)
        scheduler.step(val_loss)

        checkpoint = f'[{str(epoch + 1).rjust(len(str(manager.max_epochs)), "0")}]'
        checkpoint += f' Training PPL = {math.exp(train_loss):.16f}'
        checkpoint += f' | Validation PPL = {math.exp(val_loss):.16f}'
        checkpoint += f' | Learning Rate = {optimizer.param_groups[0]["lr"]:.16f}'
        checkpoint += f' | Elapsed Time = {elapsed}'
        logger.info(checkpoint)
        print()

        if val_loss < best_loss:
            manager.save_model((epoch, val_loss), optimizer, scheduler)
            patience, best_loss = 0, val_loss
        else:
            patience += 1

        if optimizer.param_groups[0]['lr'] < manager.min_lr:
            logger.info('Reached Minimum Learning Rate.')
            break
        if patience >= manager.max_patience:
            logger.info('Reached Maximum Patience.')
            break
        epoch += 1
    else:
        logger.info('Maximum Number of Epochs Reached.')


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pair', required=True, help='source-target language pair')
    parser.add_argument(
        '--train-data', metavar='FILE_PATH', required=True, help='parallel training data'
    )
    parser.add_argument(
        '--val-data', metavar='FILE_PATH', required=True, help='parallel validation data'
    )
    parser.add_argument('--lem-train', metavar='FILE_PATH', help='lemmatized training data')
    parser.add_argument('--lem-val', metavar='FILE_PATH', help='lemmatized validation data')
    parser.add_argument('--dict', metavar='FILE_PATH', help='bilingual dictionary')
    parser.add_argument('--freq', metavar='FILE_PATH', help='frequency statistics')
    parser.add_argument('--sw-vocab', metavar='FILE_PATH', required=True, help='subword vocab')
    parser.add_argument('--sw-model', metavar='FILE_PATH', required=True, help='subword model')
    parser.add_argument('--model', metavar='FILE_PATH', required=True, help='translation model')
    parser.add_argument('--log', metavar='FILE_PATH', required=True, help='logger output')
    parser.add_argument('--seed', type=int, help='random seed')
    args, unknown = parser.parse_known_args()

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    src_lang, tgt_lang = args.lang_pair.split('-')
    with open('translation/config.toml', 'rb') as config_file:
        config = tomllib.load(config_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        src_lang,
        tgt_lang,
        args.model,
        args.sw_vocab,
        args.sw_model,
        args.dict,
        args.freq,
    )
    train_data = manager.load_data(args.train_data, args.lem_train)
    val_data = manager.load_data(args.val_data, args.lem_val)

    if device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')

    logger = logging.getLogger('translation.logger')
    logger.addHandler(logging.FileHandler(args.log))
    logger.setLevel(logging.INFO)

    train_model(train_data, val_data, manager, logger)


if __name__ == '__main__':
    main()
