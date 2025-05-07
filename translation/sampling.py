from typing import TYPE_CHECKING

import torch
from torch import Tensor

from translation.decoder import triu_mask

if TYPE_CHECKING:
    from translation.manager import Manager


def sampling_search(
    manager: 'Manager', src_encs: Tensor, sampling_method: str, max_length: int = 512
) -> tuple[Tensor, Tensor]:
    model, vocab, device = manager.model, manager.vocab, manager.device
    tgt_mask = triu_mask(max_length, device=device)
    path = torch.full((1, max_length), vocab.BOS, device=device)
    seq_prob = torch.tensor(0.0, device=device)

    for i in range(1, max_length):
        tgt_encs = model.decode(src_encs, path[:, :i], tgt_mask=tgt_mask[:, :i, :i])
        logits = model.out_embed(tgt_encs[:, -1], inverse=True)[:, : vocab.size()]
        probabilities = logits.softmax(dim=-1).squeeze(0)

        match sampling_method:
            case 'top-k':
                path[0, i], token_prob = top_k_sampling(probabilities, manager.k)
            case 'top-p' | 'nucleus':
                path[0, i], token_prob = top_p_sampling(probabilities, manager.p)
            case 'epsilon':
                path[0, i], token_prob = epsilon_sampling(probabilities, manager.epsilon)
            case 'eta':
                path[0, i], token_prob = eta_sampling(probabilities, manager.epsilon)

        seq_prob += torch.log(torch.tensor(token_prob))
        if path[0, i] == vocab.EOS:
            break

    return path[0], seq_prob


def top_k_sampling(probabilities: Tensor, k: int) -> tuple[int, float]:
    """
    Args:
        probabilities (torch.Tensor): Probability distribution over vocabulary.
        k (int): Number of top words to keep.

    Returns:
        int: The sampled token index.
        float: The sampled token probability.
    """
    sorted_probs, sorted_idx = torch.sort(probabilities, descending=True)
    top_k_indices = sorted_idx[:k]  # Select the top-k indices

    next_token = torch.multinomial(probabilities[top_k_indices], num_samples=1)
    return int(top_k_indices[next_token].item()), float(sorted_probs[next_token].item())


def top_p_sampling(probabilities: Tensor, p: int) -> tuple[int, float]:
    """
    Args:
        probabilities (torch.Tensor): Probability distribution over vocabulary.
        p (float): Probability threshold for truncation.

    Returns:
        int: The sampled token index.
        float: The sampled token probability.
    """
    sorted_probs, sorted_idx = torch.sort(probabilities, descending=True)
    cumulative_sum = torch.cumsum(sorted_probs, 0)

    # Find the first index where cumulative probability exceeds p
    geq_p = (cumulative_sum >= p).nonzero(as_tuple=True)[0][0]

    # Select indices up to the cutoff
    real_i = sorted_idx[: geq_p + 1]

    # Sample from the allowed subset
    next_token = torch.multinomial(probabilities[real_i], num_samples=1)
    return int(real_i[next_token].item()), float(sorted_probs[next_token].item())


def epsilon_sampling(probabilities: Tensor, epsilon: float) -> tuple[int, float]:
    """
    Args:
        probabilities (torch.Tensor): Probability distribution over vocabulary.
        epsilon (float): Probability threshold for truncation.

    Returns:
        int: The sampled token index.
        float: The sampled token probability.
    """
    allowed_indices = (probabilities > epsilon).nonzero(as_tuple=True)[0]

    if len(allowed_indices) == 0:
        allowed_indices = torch.argmax(probabilities, keepdim=True)  # Fallback to max prob token

    next_token = torch.multinomial(probabilities[allowed_indices], num_samples=1)
    return int(allowed_indices[next_token].item()), float(probabilities[next_token].item())


def eta_sampling(probabilities: Tensor, epsilon: float) -> tuple[int, float]:
    """
    Args:
        probabilities (torch.Tensor): Probability distribution over vocabulary.
        epsilon (float): Base probability threshold.

    Returns:
        int: The sampled token index.
        float: The sampled token probability.
    """
    probabilities = probabilities.clamp(min=1e-9)  # Avoid log(0) issues
    entropy = -torch.sum(probabilities * torch.log(probabilities))

    eta = min(epsilon, (epsilon**0.5) * torch.exp(-entropy))
    allowed_indices = (probabilities > eta).nonzero(as_tuple=True)[0]

    if len(allowed_indices) == 0:
        allowed_indices = torch.argmax(probabilities, keepdim=True)  # Fallback to max prob token

    next_token = torch.multinomial(probabilities[allowed_indices], num_samples=1)
    return int(allowed_indices[next_token].item()), float(probabilities[next_token].item())
