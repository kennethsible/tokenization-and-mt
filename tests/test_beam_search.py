import math

import pytest
import torch


def greedy_search(word_probs, max_length):
    path = torch.full((1, max_length), -1)
    prob = 0.0

    for i in range(1, max_length):
        path[0, i] = word_probs[i - 1].argmax(dim=-1)
        prob += word_probs[i - 1, path[0, i]]

    return prob.item(), path[0, 1:].tolist()


def beam_search(word_probs, beam_size, max_length):
    vocab_size = max_length - 1
    active = torch.ones(beam_size, dtype=torch.bool)
    paths = torch.full((beam_size, max_length), -1)
    probs = torch.zeros(beam_size)

    i, init_size = 0, beam_size
    while (i := i + 1) < max_length and beam_size > 0:
        scores = probs[active].unsqueeze(1) + word_probs[i - 1]
        if i == 1:
            scores = scores[0]

        topv, topi = torch.topk(scores.flatten(), beam_size)
        if beam_size < init_size:
            active[~active] |= probs[~active] < topv.max()
            active_count = int(active.count_nonzero())
            if active_count > beam_size:
                beam_size = active_count
                topv, topi = torch.topk(scores.flatten(), beam_size)

        reorder = topi // vocab_size
        paths[active] = paths[active][reorder]
        paths[active, i] = topi % vocab_size
        probs[active] = topv

        beam_size = int(active.count_nonzero())

    return probs.tolist(), paths[:, 1:].tolist()


@pytest.fixture
def word_probs():
    return torch.tensor(
        [
            [-0.1, -0.2, -2.0],  # First Word
            [-0.3, -1.5, -0.4],  # Second Word
            [-1.2, -0.8, -0.5],  # Third Word
        ]
    )


def test_greedy_search(word_probs):
    max_length = word_probs.size(0) + 1
    expected_prob, expected_path = (-0.1 + -0.3 + -0.5, [0, 0, 2])
    actual_prob, actual_path = greedy_search(word_probs, max_length)

    assert math.isclose(actual_prob, expected_prob, rel_tol=1e-6)
    assert actual_path == expected_path


def test_beam_search_size_1(word_probs):
    max_length = word_probs.size(0) + 1
    expected_prob, expected_path = (-0.1 + -0.3 + -0.5, [0, 0, 2])
    actual_probs, actual_paths = beam_search(word_probs, 1, max_length)

    assert math.isclose(actual_probs[0], expected_prob, rel_tol=1e-6)
    assert actual_paths[0] == expected_path


def test_beam_search_size_3(word_probs):
    max_length = word_probs.size(0) + 1
    expected_sequences = [
        (-0.1 + -0.3 + -0.5, [0, 0, 2]),  # Best Sequence
        (-0.2 + -0.3 + -0.5, [1, 0, 2]),  # Second Best
        (-0.1 + -0.4 + -0.5, [0, 2, 2]),  # Third Best
    ]
    actual_sequences = beam_search(word_probs, 3, max_length)

    for actual_sequence, expected_sequence in zip(zip(*actual_sequences), expected_sequences):
        actual_prob, actual_path = actual_sequence
        expected_prob, expected_path = expected_sequence

        assert math.isclose(actual_prob, expected_prob, rel_tol=1e-6)
        assert actual_path == expected_path
