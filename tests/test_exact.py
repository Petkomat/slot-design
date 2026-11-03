"""The exact path: the reference reproduces a hand-checkable toy game, the compiled enumerator
matches the reference histogram for histogram, and the pinned games' headline figures stay put."""

from fractions import Fraction

import numpy as np
import pytest
from small_games import small_eight_symbol_game, small_three_symbol_game

from slotmath.exact import enumerate_spins, exact_result, rtp_float
from slotmath.game import UNITS_PER_LINE_BET, three_symbol_game, toy_game, uniform_weights
from slotmath.reference import enumerate_exact, play


def test_reference_reproduces_the_toy_game():
    g = toy_game()
    h = enumerate_exact(g)
    total = sum(h["pay"].values())
    rtp = Fraction(sum(x * w for x, w in h["pay"].items()), total * g.stake_units)
    assert rtp == Fraction(43637, 12800)
    assert Fraction(total - h["pay"][0], total) == Fraction(383, 512)
    assert len(h["pay"]) == 134
    pay, base, steps, runaway, scatter = play((1, 5, 3), g)
    assert (pay // UNITS_PER_LINE_BET, steps, runaway, scatter) == (79, 4, False, 0)


@pytest.mark.parametrize("game", [
    toy_game(),
    small_three_symbol_game(),
    small_three_symbol_game(verticals=True),
    small_three_symbol_game(wilds=False, ladder=(1,)),
    small_three_symbol_game(cascades=False),
    small_eight_symbol_game(),
    small_eight_symbol_game(wilds=False),
    small_eight_symbol_game(scatters=False),
    small_eight_symbol_game(cascades=False),
], ids=["toy", "three", "three+verticals", "three-no-wilds", "three-no-cascades",
        "eight", "eight-no-wilds", "eight-no-scatters", "eight-no-cascades"])
def test_compiled_enumerator_matches_reference_exactly(game):
    ex = exact_result(enumerate_spins(game))
    ref = enumerate_exact(game)
    assert ex.pay == ref["pay"]
    assert ex.base == ref["base"]
    assert ex.steps == ref["steps"]
    assert ex.scatter_pay == ref["scatter"]
    assert ex.runaways == ref["runaway"]


def test_weights_are_applied_after_enumeration():
    """One enumeration serves every weighting: uniform weights through exact_result agree with
    a game built with uniform weights from the start."""
    g = small_three_symbol_game()
    spins = enumerate_spins(g)
    uniform_game = g.with_options(weights=tuple((683, 683, 682, 683, 682, 683) for _ in range(5)))
    direct = exact_result(enumerate_spins(uniform_game))
    via = exact_result(spins, np.array(uniform_game.weights))
    assert direct.rtp == via.rtp
    assert direct.steps == via.steps


def test_three_symbol_game_headline_figures():
    ex = exact_result(enumerate_spins(three_symbol_game()))
    assert ex.rtp == Fraction(22136092935839664877, 23058430092136939520)
    assert abs(float(ex.rtp) - 0.96) < 1e-8
    assert ex.runaways == 0
    # E[D] equals the sum of the tail probabilities, exactly
    tail = sum(ex.p_steps(k) for d in range(1, max(ex.steps) + 1) for k in ex.steps if k >= d)
    assert tail == ex.expected_steps
    # the float contraction used by the tuner agrees with the exact figure
    spins = enumerate_spins(three_symbol_game())
    assert abs(rtp_float(spins, np.array(three_symbol_game().weights)) - float(ex.rtp)) < 1e-12


def test_rtp_splits_into_base_cascades_and_scatter():
    ex = exact_result(enumerate_spins(small_eight_symbol_game()))
    assert ex.scatter_rtp > 0 and ex.cascade_rtp > 0
    assert ex.base_rtp + ex.cascade_rtp + ex.scatter_rtp == ex.rtp


def test_game_validation():
    g = three_symbol_game()
    with pytest.raises(ValueError, match="sum to"):
        g.with_options(weights=uniform_weights(32, 4) + ((1,) * 32,))
    with pytest.raises(ValueError, match="symbols"):
        g.with_options(strips=("X" * 32,) * 5)
    with pytest.raises(ValueError, match="payline"):
        g.with_options(lines=((0, 1, 2, 3, 0),))
    with pytest.raises(ValueError, match="fifths"):
        g.with_options(vertical_pay={"A": Fraction(1, 3)})
    with pytest.raises(ValueError, match="cannot carry"):
        g.with_options(line_pay={**g.line_pay, "W": {3: 1}})
