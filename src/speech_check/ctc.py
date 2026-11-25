"""CTC helpers: greedy decoding and forced alignment of an expected phone sequence."""
from __future__ import annotations

import numpy as np

NEG_INF = -1e30


def forced_align(emissions: np.ndarray, labels: list[str]) -> list[tuple[int, int]] | None:
    """Viterbi CTC alignment.

    `emissions` is (frames, 1 + len(labels)) log-probabilities: column 0 is the blank, column k the
    k-th expected phone. `labels` gives the phone identity per column, since the CTC rule that an
    equal repeated label needs a blank between them depends on identity, not on position.
    Returns one (first_frame, last_frame + 1) span per label, or None when the audio is too short to
    hold the sequence.
    """
    n_frames, n_labels = emissions.shape[0], len(labels)
    if n_labels == 0:
        return []
    n_states = 2 * n_labels + 1
    repeats = sum(1 for a, b in zip(labels, labels[1:]) if a == b)
    if n_frames < n_labels + repeats:
        return None
    column = np.zeros(n_states, dtype=np.int64)  # emission column of each state
    column[1::2] = np.arange(1, n_labels + 1)
    skip_ok = np.zeros(n_states, dtype=bool)  # state s may be entered from s-2
    for k in range(1, n_labels):
        if labels[k] != labels[k - 1]:
            skip_ok[2 * k + 1] = True

    score = np.full(n_states, NEG_INF)
    score[0] = emissions[0, 0]
    score[1] = emissions[0, 1]
    back = np.zeros((n_frames, n_states), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    for t in range(1, n_frames):
        stay = score
        prev1 = np.concatenate(([NEG_INF], score[:-1]))
        prev2 = np.concatenate(([NEG_INF, NEG_INF], score[:-2]))
        prev2 = np.where(skip_ok, prev2, NEG_INF)
        stacked = np.stack((stay, prev1, prev2))
        choice = stacked.argmax(axis=0)
        back[t] = choice
        score = stacked.max(axis=0) + emissions[t, column]
    state = n_states - 1 if score[n_states - 1] >= score[n_states - 2] else n_states - 2
    first = [None] * n_labels
    last = [0] * n_labels
    for t in range(n_frames - 1, -1, -1):
        if state % 2 == 1:
            k = state // 2
            if first[k] is None or t < first[k]:
                first[k] = t
            last[k] = max(last[k], t)
        state -= int(back[t, state])
    return [(f if f is not None else 0, l + 1) for f, l in zip(first, last)]


def greedy_decode(log_probs: np.ndarray, vocab: tuple[str, ...], blank: int) -> list[dict]:
    """Collapse the frame-wise best path into tokens with frame span and peak posterior."""
    ids = log_probs.argmax(axis=1)
    out, start = [], 0
    for end in range(1, len(ids) + 1):
        if end < len(ids) and ids[end] == ids[start]:
            continue
        token = vocab[ids[start]]
        if ids[start] != blank and not (token.startswith("<") and token.endswith(">")):
            out.append({"token": token, "start": start, "end": end,
                        "confidence": float(np.exp(log_probs[start:end, ids[start]].max()))})
        start = end
    return out
