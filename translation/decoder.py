from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from translation.manager import Manager


def triu_mask(size: int, device: str | None = None) -> Tensor:
    mask = torch.ones((1, size, size), device=device)
    return torch.triu(mask, diagonal=1) == 0


def greedy_search(manager: 'Manager', src_encs: Tensor, max_length: int = 512) -> Tensor:
    model, vocab, device = manager.model, manager.vocab, manager.device
    tgt_mask = triu_mask(max_length, device=device)
    path = torch.full((1, max_length), vocab.BOS, device=device)

    for i in range(1, max_length):
        tgt_encs = model.decode(src_encs, path[:, :i], tgt_mask=tgt_mask[:, :i, :i])
        logits = model.out_embed(tgt_encs[:, -1], inverse=True)[:, : vocab.size()]
        path[0, i] = logits.log_softmax(dim=-1).argmax(dim=-1)
        if path[0, i] == vocab.EOS:
            return path[0, : i + 1]

    return path[0]


def beam_search(
    manager: 'Manager', src_encs: Tensor, beam_size: int = 5, max_length: int = 512
) -> tuple[Tensor, Tensor, Tensor]:
    model, vocab, device = manager.model, manager.vocab, manager.device
    tgt_mask = triu_mask(max_length, device=device)
    indices = torch.ones(beam_size, dtype=torch.int, device=device) * max_length
    active = torch.ones(beam_size, dtype=torch.bool, device=device)
    paths = torch.full((beam_size, max_length), vocab.BOS, device=device)
    probs = torch.zeros(beam_size, device=device)

    i, init_size = 0, beam_size
    while (i := i + 1) < max_length and beam_size > 0:
        tgt_encs = model.decode(
            src_encs.expand(beam_size, -1, -1), paths[active, :i], tgt_mask=tgt_mask[:, :i, :i]
        )
        logits = model.out_embed(tgt_encs[:, -1], inverse=True)[:, : vocab.size()]
        scores = probs[active].unsqueeze(1) + logits.log_softmax(dim=-1)
        if i == 1:
            scores = scores[0]

        topv, topi = torch.topk(scores.flatten(), beam_size)
        if beam_size < init_size:
            active[~active] |= probs[~active] < topv.max() / i
            active_count = int(active.count_nonzero())
            if active_count > beam_size:
                beam_size = active_count
                topv, topi = torch.topk(scores.flatten(), beam_size)

        reorder = topi // vocab.size()
        paths[active] = paths[active][reorder]
        paths[active, i] = topi % vocab.size()
        probs[active] = topv

        terminated = paths[:, i] == vocab.EOS
        indices[terminated] = i
        probs[terminated] /= i
        active = active & ~terminated
        beam_size = int(active.count_nonzero())

    best_path = probs.argmax()
    trunc_path = int(indices[best_path]) + 1
    return paths[best_path, :trunc_path], paths, probs
