import pytest

from speech_check.words import make_word


@pytest.fixture
def mk():
    """mk([("text", start, end), ...]) -> list of Word with sequential indexes."""
    def build(spec):
        return [make_word(i, t, s, e, 0.99) for i, (t, s, e) in enumerate(spec)]
    return build
