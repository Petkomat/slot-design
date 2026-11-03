"""Exact distribution by enumerating every stop vector.

Strip-fed cascades are deterministic once the stops are drawn, so a spin's payout is a function
of its stop vector alone. ``enumerate_spins`` evaluates all ``L ** n`` of them once and stores
what each pays. That table does not depend on the stop weights, so any weighting can then be
applied exactly: a stop vector's probability is a product of integer weights over ``4096 ** n``,
and with ``n = 5`` that product is at most ``2 ** 60``, inside int64.

The per-spin evaluator below is compiled and written independently of ``reference.py`` and of
the Monte Carlo evaluator in ``simulate.py``; the tests require all three to agree. Where the
reference tries every paying symbol on every line, this one reads the line once: the leading
wilds, then the first other symbol, which is the only symbol that can run past the wilds.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

import numpy as np
from numba import njit, prange

from .game import WEIGHT_TOTAL, Game


@njit(cache=True)
def _window_pay(grid, lines, line_pay, vertical_pay, paying, is_paying, wild, verticals, cleared):
    """Line and vertical pay of one window in units; marks winning cells in ``cleared``."""
    n, h = grid.shape
    pay = 0
    for li in range(lines.shape[0]):
        lead = 0
        while lead < n and grid[lead, lines[li, lead]] == wild:
            lead += 1
        best_pay = 0
        best_k = 0
        first = -1
        if lead < n:
            first = grid[lead, lines[li, lead]]
            if is_paying[first]:
                k = lead + 1
                while k < n:
                    c = grid[k, lines[li, k]]
                    if c == first or c == wild:
                        k += 1
                    else:
                        break
                if line_pay[first, k] > 0:
                    best_pay = line_pay[first, k]
                    best_k = k
        if lead > 0:                      # every other symbol runs exactly over the wilds
            for i in range(paying.shape[0]):
                s = paying[i]
                if s == first:
                    continue
                p = line_pay[s, lead]
                if p > 0 and (p > best_pay or (p == best_pay and lead > best_k)):
                    best_pay = p
                    best_k = lead
        if best_pay > 0:
            pay += best_pay
            for r in range(best_k):
                cleared[r, lines[li, r]] = True
    if verticals:
        for r in range(n):
            s = grid[r, 0]
            if is_paying[s]:
                same = True
                for k in range(1, h):
                    if grid[r, k] != s:
                        same = False
                        break
                if same:
                    pay += vertical_pay[s]
                    for k in range(h):
                        cleared[r, k] = True
    return pay


@njit(cache=True)
def _play(stops, strips, lines, line_pay, vertical_pay, scatter_pay, paying, is_paying, ladder,
          flags, grid, cleared, offsets, column):
    """One spin: (total pay, step-0 line pay, cascade steps, runaway?, scatters on the final
    window). The total includes the scatter pay."""
    n, L = strips.shape
    h, wild, verticals, cascades, max_steps, scatter = (flags[0], flags[1], flags[2] == 1,
                                                        flags[3] == 1, flags[4], flags[5])
    for r in range(n):
        offsets[r] = stops[r]
        for k in range(h):
            grid[r, k] = strips[r, (stops[r] + k) % L]
    total = 0
    base = 0
    steps = 0
    runaway = False
    top = ladder.shape[0] - 1
    while True:
        cleared[:, :] = False
        pay = _window_pay(grid, lines, line_pay, vertical_pay, paying, is_paying, wild,
                          verticals, cleared)
        if pay == 0:
            break
        pay *= ladder[steps] if steps < top else ladder[top]
        total += pay
        if steps == 0:
            base = pay
        steps += 1
        if not cascades:
            break
        if steps >= max_steps:
            runaway = True
            break
        for r in range(n):
            d = 0
            for k in range(h):
                if cleared[r, k]:
                    d += 1
            if d == 0:
                continue
            new_offset = (offsets[r] - d) % L
            j = 0
            for k in range(d):
                column[j] = strips[r, (new_offset + k) % L]
                j += 1
            for k in range(h):
                if not cleared[r, k]:
                    column[j] = grid[r, k]
                    j += 1
            for k in range(h):
                grid[r, k] = column[k]
            offsets[r] = new_offset
    count = 0
    for r in range(n):
        for k in range(h):
            if grid[r, k] == scatter:
                count += 1
    return total + scatter_pay[count], base, steps, runaway, count


@njit(parallel=True, cache=True)
def _enumerate(strips, lines, line_pay, vertical_pay, scatter_pay, paying, ladder, flags):
    n, L = strips.shape
    h = flags[0]
    block = L ** (n - 1)
    pay_total = np.empty(L * block, np.int32)
    pay_base = np.empty(L * block, np.int32)
    steps_out = np.empty(L * block, np.int8)
    scatters_out = np.empty(L * block, np.int8)
    trouble = np.zeros((L, 2), np.int64)  # per first-reel stop: runaways, int32 overflows
    is_paying = np.zeros(line_pay.shape[0], np.bool_)
    for i in range(paying.shape[0]):
        is_paying[paying[i]] = True
    for t1 in prange(L):
        grid = np.empty((n, h), np.int64)
        cleared = np.empty((n, h), np.bool_)
        offsets = np.empty(n, np.int64)
        column = np.empty(h, np.int64)
        stops = np.empty(n, np.int64)
        for j in range(block):
            rest = j
            for r in range(n - 1, 0, -1):
                stops[r] = rest % L
                rest //= L
            stops[0] = t1
            total, base, steps, runaway, count = _play(
                stops, strips, lines, line_pay, vertical_pay, scatter_pay, paying, is_paying,
                ladder, flags, grid, cleared, offsets, column)
            idx = t1 * block + j
            pay_total[idx] = total
            pay_base[idx] = base
            steps_out[idx] = min(steps, 127)
            scatters_out[idx] = count
            if runaway:
                trouble[t1, 0] += 1
            if total > 2147483647:
                trouble[t1, 1] += 1
    return pay_total, pay_base, steps_out, scatters_out, trouble.sum(axis=0)


@njit(cache=True)
def _weighted_histograms(pay_total, pay_base, steps, scatters, weights, max_pay, max_base,
                         max_steps, max_scatters):
    """Exact weighted counts: a (steps, pay) table, a step-0 pay table, a scatter-count table."""
    n, L = weights.shape
    joint = np.zeros((max_steps + 1, max_pay + 1), np.int64)
    base_hist = np.zeros(max_base + 1, np.int64)
    scatter_hist = np.zeros(max_scatters + 1, np.int64)
    for idx in range(pay_total.shape[0]):
        rest = idx
        w = 1
        for r in range(n - 1, -1, -1):
            w *= weights[r, rest % L]
            rest //= L
        joint[steps[idx], pay_total[idx]] += w
        base_hist[pay_base[idx]] += w
        scatter_hist[scatters[idx]] += w
    return joint, base_hist, scatter_hist


@dataclass(frozen=True)
class Spins:
    """What every stop vector pays, independent of the weights. Arrays are in stop-vector order:
    reel 1's stop is the most significant base-``L`` digit."""

    game: Game
    pay_total: np.ndarray
    pay_base: np.ndarray
    steps: np.ndarray
    scatters: np.ndarray        # scatters visible on the final window
    runaways: int


def enumerate_spins(game: Game) -> Spins:
    """Evaluate every one of the ``L ** n`` stop vectors."""
    a = game.arrays()
    pay_total, pay_base, steps, scatters, trouble = _enumerate(
        a["strips"], a["lines"], a["line_pay"], a["vertical_pay"], a["scatter_pay"],
        a["paying"], a["ladder"], a["flags"])
    if trouble[1]:
        raise OverflowError(f"{trouble[1]} spins pay more than int32 holds")
    return Spins(game, pay_total, pay_base, steps, scatters, int(trouble[0]))


@dataclass(frozen=True)
class ExactResult:
    """Exact figures for one game. Pays are in units (a fifth of a line bet); every probability
    is an integer over ``4096 ** n``, kept as a ``Fraction``."""

    stake_units: int
    total_weight: int
    joint: dict[tuple[int, int], int]   # (cascade steps, total pay) -> weight
    base: dict[int, int]                # step-0 line pay -> weight
    scatters: dict[int, int]            # scatters on the final window -> weight
    scatter_units: tuple[int, ...]      # scatter pay by count
    runaways: int

    # -- distributions
    @property
    def pay(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for (_, x), w in self.joint.items():
            out[x] = out.get(x, 0) + w
        return out

    @property
    def steps(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for (d, _), w in self.joint.items():
            out[d] = out.get(d, 0) + w
        return out

    @property
    def scatter_pay(self) -> dict[int, int]:
        """Scatter pay in units -> weight."""
        out: dict[int, int] = {}
        for c, w in self.scatters.items():
            x = self.scatter_units[c]
            out[x] = out.get(x, 0) + w
        return out

    def probability(self, weight: int) -> Fraction:
        return Fraction(weight, self.total_weight)

    # -- headline figures
    def _mean(self, hist: dict[int, int]) -> Fraction:
        return Fraction(sum(x * w for x, w in hist.items()), self.total_weight * self.stake_units)

    @property
    def rtp(self) -> Fraction:
        return self._mean(self.pay)

    @property
    def base_rtp(self) -> Fraction:
        """RTP of the first window's line wins."""
        return self._mean(self.base)

    @property
    def scatter_rtp(self) -> Fraction:
        return self._mean(self.scatter_pay)

    @property
    def cascade_rtp(self) -> Fraction:
        """RTP of the line wins after the first window, multipliers included."""
        return self.rtp - self.base_rtp - self.scatter_rtp

    @property
    def hit_frequency(self) -> Fraction:
        return 1 - Fraction(self.pay.get(0, 0), self.total_weight)

    @property
    def variance(self) -> Fraction:
        """Variance of the payout per unit staked."""
        m1 = Fraction(sum(x * w for x, w in self.pay.items()), self.total_weight)
        m2 = Fraction(sum(x * x * w for x, w in self.pay.items()), self.total_weight)
        return (m2 - m1 * m1) / (self.stake_units ** 2)

    @property
    def volatility(self) -> float:
        """Standard deviation of the payout per unit staked."""
        v = self.variance
        return isqrt(v.numerator * 10**36 // v.denominator) / 10**18

    @property
    def max_win(self) -> Fraction:
        return Fraction(max(self.pay), self.stake_units)

    def p_win_at_least(self, multiple_of_stake: int | Fraction) -> Fraction:
        threshold = multiple_of_stake * self.stake_units
        return Fraction(sum(w for x, w in self.pay.items() if x >= threshold), self.total_weight)

    def p_win_above(self, multiple_of_stake: int | Fraction) -> Fraction:
        threshold = multiple_of_stake * self.stake_units
        return Fraction(sum(w for x, w in self.pay.items() if x > threshold), self.total_weight)

    # -- cascades
    def p_steps(self, d: int) -> Fraction:
        return Fraction(self.steps.get(d, 0), self.total_weight)

    @property
    def expected_steps(self) -> Fraction:
        """E[D], D = number of windows with a line win in a spin (0 when the lines pay nothing)."""
        return Fraction(sum(d * w for d, w in self.steps.items()), self.total_weight)

    @property
    def expected_steps_given_win(self) -> Fraction:
        """E[D | D >= 1]: chain length given that the first window has a line win."""
        won = self.total_weight - self.steps.get(0, 0)
        return Fraction(sum(d * w for d, w in self.steps.items()), won)

    def rtp_by_steps(self) -> dict[int, Fraction]:
        """RTP (scatter pay included) contributed by spins with exactly d winning windows."""
        out: dict[int, int] = {}
        for (d, x), w in self.joint.items():
            out[d] = out.get(d, 0) + x * w
        denom = self.total_weight * self.stake_units
        return {d: Fraction(v, denom) for d, v in sorted(out.items())}


def exact_result(spins: Spins, weights: np.ndarray | None = None) -> ExactResult:
    """Apply stop weights (the game's own by default) to an enumeration, exactly."""
    game = spins.game
    w = np.asarray(game.weights if weights is None else weights, dtype=np.int64)
    if (w.sum(axis=1) != WEIGHT_TOTAL).any():
        raise ValueError(f"weights must sum to {WEIGHT_TOTAL} on every reel")
    joint, base_hist, scatter_hist = _weighted_histograms(
        spins.pay_total, spins.pay_base, spins.steps, spins.scatters, w,
        int(spins.pay_total.max()), int(spins.pay_base.max()), int(spins.steps.max()),
        int(spins.scatters.max()))
    total_weight = WEIGHT_TOTAL ** game.reels
    if int(joint.sum()) != total_weight:  # every stop vector counted exactly once
        raise AssertionError("weighted counts do not sum to 4096 ** n")
    nz = np.nonzero(joint)
    joint_d = {(int(d), int(x)): int(joint[d, x]) for d, x in zip(*nz)}
    base_d = {int(x): int(base_hist[x]) for x in np.nonzero(base_hist)[0]}
    scatter_d = {int(c): int(scatter_hist[c]) for c in np.nonzero(scatter_hist)[0]}
    units = tuple(game.scatter_units(c) for c in range(game.reels * game.rows + 1))
    return ExactResult(game.stake_units, total_weight, joint_d, base_d, scatter_d, units,
                       spins.runaways)


def rtp_float(spins: Spins, weights: np.ndarray) -> float:
    """RTP in floating point, for tuning: contract the pay tensor with each reel's stop
    probabilities. Fast, and only used to search; reported figures come from ``exact_result``."""
    game = spins.game
    shape = (game.length,) * game.reels
    tensor = spins.pay_total.reshape(shape).astype(np.float64)
    probs = np.asarray(weights, dtype=np.float64) / WEIGHT_TOTAL
    for r in range(game.reels - 1, -1, -1):
        tensor = tensor @ probs[r]
    return float(tensor) / game.stake_units
