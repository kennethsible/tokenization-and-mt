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
    path = torch.full((1, max_length), vocab.BOS, device=device)

    for layer in model.decoder.layers:
        layer.self_attn.kv_cache = None
        layer.crss_attn.kv_cache = None

    for i in range(1, max_length):
        tgt_encs = model.decode(src_encs, path[:, :i], autoregressive=True)
        logits = model.out_embed(tgt_encs[:, -1], inverse=True)[:, : vocab.size()]
        path[0, i] = logits.log_softmax(dim=-1).argmax(dim=-1)
        if path[0, i] == vocab.EOS:
            return path[0, : i + 1]

    return path[0]


def beam_search(
    manager: 'Manager',
    src_encs: Tensor,
    beam_size: int = 5,
    max_length: int = 512,
    alpha: float = 0.7,
) -> Tensor:
    model, vocab, device = manager.model, manager.vocab, manager.device
    paths = torch.full((beam_size, max_length), vocab.BOS, device=device)
    probs = torch.zeros(beam_size, device=device)
    finished_paths, finished_probs = [], []

    for layer in model.decoder.layers:
        layer.self_attn.kv_cache = None
        layer.crss_attn.kv_cache = None

    i = 1
    while i < max_length and beam_size > 0:
        if i == 1:
            tgt_encs = model.decode(src_encs, paths[0, :i].unsqueeze(0), autoregressive=True)
            logits = model.out_embed(tgt_encs[:, -1], inverse=True)[:, : vocab.size()]
            scores = logits.log_softmax(dim=-1)
        else:
            tgt_encs = model.decode(
                src_encs.expand(beam_size, -1, -1), paths[:, :i], autoregressive=True
            )
            logits = model.out_embed(tgt_encs[:, -1], inverse=True)[:, : vocab.size()]
            scores = probs.unsqueeze(1) + logits.log_softmax(dim=-1)

        topv, topi = torch.topk(scores.flatten(), beam_size)
        parents = topi // vocab.size()
        if i > 1:
            paths = paths[parents]
        paths[:, i] = topi % vocab.size()
        probs = topv

        eos_mask = paths[:, i] == vocab.EOS
        active = torch.where(~eos_mask)[0]
        finished_indices = torch.where(eos_mask)[0]

        for index in finished_indices:
            length_norm = ((5 + i) ** alpha) / ((5 + 1) ** alpha)
            finished_paths.append(paths[index].clone())
            finished_probs.append(probs[index].item() / length_norm)

        for layer in model.decoder.layers:
            if layer.self_attn.kv_cache is not None:
                k_cache, v_cache = layer.self_attn.kv_cache
                if i == 1:
                    k_cache = k_cache.expand(beam_size, -1, -1, -1).clone()
                    v_cache = v_cache.expand(beam_size, -1, -1, -1).clone()
                else:
                    k_cache = k_cache[parents][active]
                    v_cache = v_cache[parents][active]
                layer.self_attn.kv_cache = (k_cache, v_cache)

        paths = paths[active]
        probs = probs[active]
        beam_size = active.numel()
        i += 1

    if finished_paths:
        return finished_paths[torch.tensor(finished_probs).argmax()]
    return paths[probs.argmax()]
