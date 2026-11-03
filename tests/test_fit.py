"""The fitter: strips keep their composition and stack heights, the float moments it searches
with agree with exact arithmetic, a single-unit weight move is priced exactly, and the fitted
designs meet their targets when recomputed exactly from the integer weights."""

from fractions import Fraction

import numpy as np
import pytest
from small_games import small_eight_symbol_game

from slotmath.exact import enumerate_spins, exact_result
from slotmath.fit import (
    LENGTH,
    RTP_TOLERANCE,
    VOLATILITY_TOLERANCE,
    WEIGHT_BOUNDS,
    Moments,
    StripParams,
    arrange,
    composition,
    solve_weights,
    strips_for,
)
from slotmath.game import HIGH, fitted_game, load_designs


def _runs(strip):
    """Lengths of the circular runs of equal symbols, as (symbol, length)."""
    n = len(strip)
    start = next(i for i in range(n) if strip[i] != strip[i - 1])
    out, i = [], 0
    while i < n:
        s, k = strip[(start + i) % n], 0
        while k < n and strip[(start + i + k) % n] == s:
            k += 1
        out.append((s, k))
        i += k
    return out


@pytest.mark.parametrize("params", [StripParams(3, 1, 1), StripParams(9, 3, 2),
                                    StripParams(17, 6, 3, seed=4)])
def test_strips_keep_composition_and_stack_height(params):
    for r, strip in enumerate(strips_for(params)):
        assert len(strip) == LENGTH
        counts = composition(params, r)
        assert {s: strip.count(s) for s in counts} == counts
        assert counts["S"] == 1 and (counts["W"] == 0) == (r in (0, 4))
        for s, k in _runs(strip):
            assert k <= (params.stack if s in HIGH else 1), (s, k)


def test_arrange_refuses_an_impossible_composition():
    """Three blocks of one symbol among four cannot be kept apart around a circle."""
    with pytest.raises(ValueError, match="cannot be kept apart"):
        arrange({"L1": 3, "L2": 1}, 1, seed=0)


def _exact_moments(game, weights):
    ex = exact_result(enumerate_spins(game), np.array(weights))
    m1 = Fraction(sum(x * w for x, w in ex.pay.items()), ex.total_weight)
    m2 = Fraction(sum(x * x * w for x, w in ex.pay.items()), ex.total_weight)
    return m1, m2


def test_float_moments_agree_with_exact_arithmetic():
    g = small_eight_symbol_game()
    mom = Moments(enumerate_spins(g).pay_total, g.length, g.reels)
    w = np.array(g.weights)
    (m1, _), (m2, _) = mom.evaluate(w / 4096)
    e1, e2 = _exact_moments(g, w)
    assert abs(m1 - float(e1)) <= 1e-12 * float(e1)
    assert abs(m2 - float(e2)) <= 1e-12 * float(e2)


def test_single_unit_move_is_priced_exactly():
    """Moving one unit of weight from stop i to stop j on reel r changes E[X] by exactly
    (G_r[j] - G_r[i]) / 4096, because E[X] is linear in reel r's weights."""
    g = small_eight_symbol_game()
    mom = Moments(enumerate_spins(g).pay_total, g.length, g.reels)
    w = np.array(g.weights)
    (_, g1), (_, g2) = mom.evaluate(w / 4096)
    before = _exact_moments(g, w)
    for r, i, j in [(0, 1, 4), (2, 5, 0), (4, 3, 2)]:
        moved = w.copy()
        moved[r, i] -= 1
        moved[r, j] += 1
        after = _exact_moments(g, moved)
        assert abs(float(after[0] - before[0]) - (g1[r][j] - g1[r][i]) / 4096) < 1e-9
        assert abs(float(after[1] - before[1]) - (g2[r][j] - g2[r][i]) / 4096) < 1e-6


def test_solver_reaches_both_targets():
    g = small_eight_symbol_game()
    mom = Moments(enumerate_spins(g).pay_total, g.length, g.reels)
    base = exact_result(enumerate_spins(g))
    rtp, vol = 0.95 * float(base.rtp), 1.05 * base.volatility
    sol = solve_weights(mom, g.stake_units, rtp, vol)
    assert sol.feasible
    assert abs(sol.rtp - rtp) < 1e-10 and abs(sol.volatility - vol) < 1e-8 * vol


def _check_design(key):
    designs = load_designs()
    d = designs["games"][key]
    game = fitted_game(key)
    assert [list(s) for s in strips_for(StripParams(**d["params"]))] == d["strips"]
    w = np.array(game.weights)
    assert w.min() >= WEIGHT_BOUNDS[0] and w.max() <= WEIGHT_BOUNDS[1]
    ex = exact_result(enumerate_spins(game))
    assert f"{ex.rtp.numerator}/{ex.rtp.denominator}" == d["rtp_exact"]
    assert abs(float(ex.rtp) - designs["inputs"]["rtp"]) <= RTP_TOLERANCE
    target = d["target_volatility"]
    assert abs(ex.volatility - target) <= VOLATILITY_TOLERANCE * target
    assert ex.runaways == 0


def test_main_design_meets_its_targets():
    _check_design(load_designs()["main"])


@pytest.mark.slow
@pytest.mark.parametrize("key", ["v1", "v2", "v3", "v4", "v5"])
def test_every_design_meets_its_targets(key):
    _check_design(key)
