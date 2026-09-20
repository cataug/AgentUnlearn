from agentunlearn.scoring import exact_match, token_f1
from agentunlearn.stats import mcnemar_exact, wilson_interval


def test_exact_match_normalized():
    assert exact_match("Hello, world!", "hello world") == 1.0


def test_token_f1():
    assert token_f1("a b c", "a b") > 0.7
    assert token_f1("x", "y") == 0.0


def test_wilson():
    lo, hi = wilson_interval(5, 10)
    assert 0 <= lo < 0.5 < hi <= 1


def test_mcnemar():
    out = mcnemar_exact([0, 0, 1, 1], [1, 0, 1, 0])
    assert out["discordant_01"] == 1
    assert out["discordant_10"] == 1
