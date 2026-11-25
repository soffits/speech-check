import numpy as np

from speech_check.ctc import forced_align, greedy_decode


def emissions(path, n_cols):
    """Frame-wise one-hot-ish log emissions for a path of column ids."""
    e = np.full((len(path), n_cols), np.log(0.01))
    for t, c in enumerate(path):
        e[t, c] = np.log(0.96)
    return e


def test_forced_align_recovers_spans():
    # columns: 0 blank, 1 'a', 2 'b'; path a a _ b b b
    e = emissions([1, 1, 0, 2, 2, 2], 3)
    assert forced_align(e, ["a", "b"]) == [(0, 2), (3, 6)]


def test_forced_align_repeated_label_needs_blank():
    e = emissions([1, 0, 2], 3)
    assert forced_align(e, ["a", "a"]) == [(0, 1), (2, 3)]
    assert forced_align(emissions([1, 1], 3), ["a", "a"]) is None  # too short for a doubled label


def test_forced_align_too_short_and_empty():
    assert forced_align(emissions([1, 2], 4), ["a", "b", "c"]) is None
    assert forced_align(emissions([0], 2), []) == []


def test_greedy_decode_collapses_and_skips_blank_and_specials():
    vocab = ("<pad>", "<s>", "a", "b")
    lp = np.log(np.array([[.1, .0, .9, 0], [.1, 0, .9, 0], [.9, 0, .1, 0], [.1, 0, 0, .9], [.1, .8, 0, 0]]) + 1e-9)
    toks = greedy_decode(lp, vocab, 0)
    assert [t["token"] for t in toks] == ["a", "b"]
    assert toks[0]["start"] == 0 and toks[0]["end"] == 2
