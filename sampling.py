import torch

def top_k_sampling(probabilities, k):
    """
    Args:
        probabilities (torch.Tensor): Probability distribution over vocabulary.
        k (int): Number of top words to keep.

    Returns:
        int: The sampled token index.
    """
    sorted_probs, sorted_idx = torch.sort(probabilities, descending=True)
    top_k_indices = sorted_idx[:k]  # Select the top-k indices
    
    next_token = torch.multinomial(probabilities[top_k_indices], num_samples=1)
    return top_k_indices[next_token].item()

def top_p_sampling(probabilities, p):
    """
    Args:
        probabilities (torch.Tensor): Probability distribution over vocabulary.
        p (float): Probability threshold for truncation.

    Returns:
        int: The sampled token index.
    """
    sorted_probs, sorted_idx = torch.sort(probabilities, descending=True)
    cumulative_sum = torch.cumsum(sorted_probs, 0)
    
    # Find the first index where cumulative probability exceeds p
    geq_p = (cumulative_sum >= p).nonzero(as_tuple=True)[0][0]
    
    # Select indices up to the cutoff
    real_i = sorted_idx[:geq_p + 1]
    
    # Sample from the allowed subset
    next_token = torch.multinomial(probabilities[real_i], num_samples=1)
    return real_i[next_token].item()

def epsilon_sampling(probabilities, epsilon):
    """
    Args:
        probabilities (torch.Tensor): Probability distribution over vocabulary.
        epsilon (float): Probability threshold for truncation.

    Returns:
        int: The sampled token index.
    """
    allowed_indices = (probabilities > epsilon).nonzero(as_tuple=True)[0]
    
    if len(allowed_indices) == 0:
        allowed_indices = torch.argmax(probabilities, keepdim=True)  # Fallback to max prob token
    
    next_token = torch.multinomial(probabilities[allowed_indices], num_samples=1)
    return allowed_indices[next_token].item()

def eta_sampling(probabilities, epsilon):
    """
    Args:
        probabilities (torch.Tensor): Probability distribution over vocabulary.
        epsilon (float): Base probability threshold.

    Returns:
        int: The sampled token index.
    """
    probabilities = probabilities.clamp(min=1e-9)  # Avoid log(0) issues
    entropy = -torch.sum(probabilities * torch.log(probabilities))
    
    eta = min(epsilon, (epsilon ** 0.5) * torch.exp(-entropy))
    allowed_indices = (probabilities > eta).nonzero(as_tuple=True)[0]
    
    if len(allowed_indices) == 0:
        allowed_indices = torch.argmax(probabilities, keepdim=True)  # Fallback to max prob token
    
    next_token = torch.multinomial(probabilities[allowed_indices], num_samples=1)
    return allowed_indices[next_token].item()
