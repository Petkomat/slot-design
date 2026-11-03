"""The Monte Carlo path: its random numbers, its sampler and its evaluator each match a
pure-Python twin exactly, it reproduces regardless of thread count, and it agrees with the exact
path statistically. The two compiled evaluators also agree on every stop vector of the fitted
games, which leaves sampling as the only source of difference between the paths."""

import itertools
import random

import numba
import numpy as np
import pytest
from small_games import small_eight_symbol_game

from slotmath.exact import enumerate_spins, exact_result
from slotmath.game import fitted_game, load_designs, three_symbol_game, toy_game
from slotmath.reference import play
from slotmath.simulate import (
    _next,
    _stream_state,
    checkpoints_geometric,
    evaluate_every_stop,
    kernel_arrays,
    lookup_tables,
    simulate,
    simulate_python,
    spin_args,
    spin_once,
    splitmix64_python,
    stream_state_python,
)


def test_splitmix64_matches_python_twin():
    state_c = state_py = 123456789
    for _ in range(1000):
        # keep the state uint64 across the Python boundary: numba hands it back as a Python
        # int, and an int64 state would make `state + gamma` a float inside the kernel
        state_c, out_c = _next(np.uint64(state_c))
        state_py, out_py = splitmix64_python(state_py)
        assert int(out_c) == out_py
        assert int(state_c) == state_py
    for seed, stream in [(0, 0), (2026, 7), (2**40 + 3, 99_999)]:
        assert int(_stream_state(seed, stream)) == stream_state_python(seed, stream)


def test_lookup_table_holds_each_stop_weight_times():
    w = np.array(three_symbol_game().weights)
    lut = lookup_tables(w)
    assert lut.shape == (5, 4096)
    for r in range(5):
        assert (np.bincount(lut[r], minlength=32) == w[r]).all()


OPTION_SETS = [toy_game()] + [
    three_symbol_game(wilds=w, verticals=v, cascades=c, ladder=lad)
    for w, v, c, lad in itertools.product((True, False), (True, False), (True, False),
                                        ((1, 2, 3, 5), (1,)))] + [
    small_eight_symbol_game(wilds=w, scatters=s, cascades=c)
    for w, s, c in itertools.product((True, False), (True, False), (True, False))]


@pytest.mark.parametrize("game", OPTION_SETS)
def test_evaluator_matches_reference_on_random_spins(game):
    rng = random.Random(7)
    a, ra = kernel_arrays(game), game.arrays()
    for _ in range(500):
        stops = tuple(rng.randrange(game.length) for _ in range(game.reels))
        total, base, steps, runaway, scatter = spin_once(np.array(stops), *spin_args(a))
        assert (total, base, steps, bool(runaway), scatter) == play(stops, game, ra), stops


@pytest.mark.parametrize("game", [three_symbol_game(), small_eight_symbol_game()],
                         ids=["three", "eight"])
def test_compiled_session_is_bit_identical_to_python_twin(game):
    sim = simulate(game, 1, 5000, seed=11)
    twin = simulate_python(game, 5000, seed=11)
    assert int(sim.pay_sum[0]) == twin["pay_sum"]
    assert int(sim.pay_sq[0]) == twin["pay_sq"]
    assert int(sim.base_sum[0]) == twin["base_sum"]
    assert int(sim.scatter_sum[0]) == twin["scatter_sum"]
    assert int(sim.hits[0]) == twin["hits"]


def _agree_everywhere(key):
    """The two compiled evaluators, written independently, on every one of the L ** n stop
    vectors: equal totals and equal chain lengths, no sampling involved."""
    game = fitted_game(key)
    exact = enumerate_spins(game)
    total, steps = evaluate_every_stop(game)
    assert (total == exact.pay_total).all()
    assert (steps == exact.steps).all()


def test_evaluators_agree_on_every_stop_of_the_main_game():
    _agree_everywhere(load_designs()["main"])


@pytest.mark.slow
@pytest.mark.parametrize("key", ["v1", "v2", "v3", "v4", "v5"])
def test_evaluators_agree_on_every_stop_of_every_design(key):
    _agree_everywhere(key)


def test_result_does_not_depend_on_thread_count():
    # as many threads as Numba allows here, up to 4: a 2-core CI runner allows 2
    threads = min(4, numba.config.NUMBA_NUM_THREADS)
    if threads < 2:
        pytest.skip("needs at least two threads")
    g = three_symbol_game()
    one = simulate(g, 24, 2000, seed=5, threads=1)
    many = simulate(g, 24, 2000, seed=5, threads=threads)
    assert (one.pay_sum == many.pay_sum).all()
    assert (one.cum == many.cum).all()


def test_geometric_checkpoints():
    ck = checkpoints_geometric(100)
    assert list(ck) == [1, 2, 3, 5, 8, 11, 17, 26, 38, 58, 86]   # round(1.5**2) = 2 is dropped


@pytest.mark.parametrize("game", [toy_game(), small_eight_symbol_game()], ids=["toy", "eight"])
def test_monte_carlo_agrees_with_exact(game):
    ex = exact_result(enumerate_spins(game))
    sim = simulate(game, 20, 100_000, seed=3)
    assert abs(sim.rtp - float(ex.rtp)) / sim.rtp_se < 4
    assert abs(sim.expected_steps - float(ex.expected_steps)) / sim.expected_steps_se < 4


@pytest.mark.slow
def test_monte_carlo_agrees_with_exact_on_the_three_symbol_game():
    g = three_symbol_game()
    ex = exact_result(enumerate_spins(g))
    sim = simulate(g, 100, 1_000_000, seed=9, big_wins=(10, 100))
    assert abs(sim.rtp - float(ex.rtp)) / sim.rtp_se < 4
    assert abs(sim.expected_steps - float(ex.expected_steps)) / sim.expected_steps_se < 4
    for i, m in enumerate((10, 100)):
        p = float(ex.p_win_at_least(m))
        se = (p * (1 - p) / sim.spins) ** 0.5
        assert abs(sim.p_win_at_least()[i] - p) / se < 4
