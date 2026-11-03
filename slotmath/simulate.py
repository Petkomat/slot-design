"""Monte Carlo simulation (compiled), written independently of ``exact.py``.

The evaluator here deliberately uses different algorithms from the exact path: a payline is
scanned once with a shrinking bitmask of the symbols still matching it, winning cells are a
64-bit mask over the flattened window, and a refilled column is rebuilt from the bottom up.
The tests require it to agree with ``reference.py`` on every spin they try.

Efficiency:

* Stop weights sum to 4096 = 2**12 per reel, and each reel has a 4096-entry lookup table from
  12 random bits to a stop, so sampling is one table read with no modulo bias.
* One 64-bit splitmix64 draw supplies five reels' worth of 12-bit stops.
* Nothing is allocated inside the spin loop; each session owns its scratch arrays.
* Sessions run in parallel (``prange``). Each is its own random stream, seeded from the master
  seed and the session index, so a run reproduces bit for bit whatever the thread count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import sqrt

import numba
import numpy as np
from numba import njit, prange

from .game import WEIGHT_TOTAL, Game
from .reference import play as reference_play

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_LOW12 = np.uint64(0xFFF)
_MASK64 = (1 << 64) - 1


# ------------------------------------------------------------------- random numbers
@njit(cache=True)
def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def _next(state):
    state = state + _GOLDEN
    return state, _mix(state)


@njit(cache=True)
def _stream_state(seed, stream):
    return _mix(_mix(np.uint64(seed) + _GOLDEN * np.uint64(stream + 1)))


def splitmix64_python(state: int) -> tuple[int, int]:
    """Pure-Python twin of ``_next``: (new state, output)."""
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return state, z ^ (z >> 31)


def stream_state_python(seed: int, stream: int) -> int:
    def mix(z: int) -> int:
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)
    return mix(mix((seed + 0x9E3779B97F4A7C15 * (stream + 1)) & _MASK64))


def lookup_tables(weights) -> np.ndarray:
    """Per reel, 4096 entries: stop t appears ``weights[r][t]`` times."""
    w = np.asarray(weights, dtype=np.int64)
    if (w.sum(axis=1) != WEIGHT_TOTAL).any():
        raise ValueError(f"weights must sum to {WEIGHT_TOTAL} on every reel")
    return np.stack([np.repeat(np.arange(w.shape[1]), row) for row in w]).astype(np.int64)



# ------------------------------------------------------------------- the evaluator
@njit(cache=True)
def _evaluate(g, lines_flat, line_pay, vertical_pay, slot_symbol, match, n, h, verticals, masks):
    """(line and vertical pay in units, winning-cell bitmask) for the flattened window ``g``
    (cell r*h + k). ``match[c]`` is the set of paying slots cell symbol ``c`` counts as: one bit
    for a paying symbol, every bit for the wild, none otherwise."""
    pay = 0
    hit = np.int64(0)
    for li in range(lines_flat.shape[0]):
        alive = match[g[lines_flat[li, 0]]]
        if alive == 0:
            continue
        masks[0] = alive
        depth = 1                              # masks[r]: slots matching cells 0..r
        while depth < n:
            alive &= match[g[lines_flat[li, depth]]]
            if alive == 0:
                break
            masks[depth] = alive
            depth += 1
        best_p = 0
        best_k = 0
        for i in range(slot_symbol.shape[0]):
            bit = np.int64(1) << i
            if (masks[0] & bit) == 0:
                continue
            k = 1
            while k < depth and masks[k] & bit:
                k += 1
            p = line_pay[slot_symbol[i], k]
            if p > 0 and (p > best_p or (p == best_p and k > best_k)):
                best_p = p
                best_k = k
        if best_p > 0:
            pay += best_p
            for r in range(best_k):
                hit |= np.int64(1) << lines_flat[li, r]
    if verticals:
        for r in range(n):
            s = g[r * h]
            if line_pay[s].max() > 0:
                k = 1
                while k < h and g[r * h + k] == s:
                    k += 1
                if k == h:
                    pay += vertical_pay[s]
                    hit |= ((np.int64(1) << h) - 1) << (r * h)
    return pay, hit


@njit(cache=True)
def _spin(stops, strips, lines_flat, line_pay, vertical_pay, scatter_pay, slot_symbol, match,
          ladder, flags, g, off, tmp, masks):
    """One spin: (total pay, step-0 line pay, cascade steps, runaway 0/1, scatter pay). The
    total includes the scatter pay."""
    n, L = strips.shape
    h = flags[0]
    verticals = flags[2] == 1
    cascades = flags[3] == 1
    max_steps = flags[4]
    scatter = flags[5]
    for r in range(n):
        off[r] = stops[r]
        for k in range(h):
            g[r * h + k] = strips[r, (stops[r] + k) % L]
    total = 0
    base = 0
    steps = 0
    runaway = 0
    last = ladder.shape[0] - 1
    while True:
        pay, hit = _evaluate(g, lines_flat, line_pay, vertical_pay, slot_symbol, match, n, h,
                             verticals, masks)
        if pay == 0:
            break
        pay *= ladder[min(steps, last)]
        total += pay
        if steps == 0:
            base = pay
        steps += 1
        if not cascades:
            break
        if steps >= max_steps:
            runaway = 1
            break
        for r in range(n):
            if ((hit >> (r * h)) & ((np.int64(1) << h) - 1)) == 0:
                continue
            write = h - 1                      # survivors sink to the bottom, in order
            for k in range(h - 1, -1, -1):
                if not (hit >> (r * h + k)) & 1:
                    tmp[write] = g[r * h + k]
                    write -= 1
            d = write + 1                      # cleared cells, refilled from above
            off[r] = (off[r] - d) % L
            for k in range(d):
                tmp[k] = strips[r, (off[r] + k) % L]
            for k in range(h):
                g[r * h + k] = tmp[k]
    seen = 0
    for c in range(n * h):
        if g[c] == scatter:
            seen += 1
    return total + scatter_pay[seen], base, steps, runaway, scatter_pay[seen]


@njit(cache=True)
def _draw_stops(state, lut, stops):
    n = stops.shape[0]
    state, z = _next(state)
    used = 0
    for r in range(n):
        if used == 5:
            state, z = _next(state)
            used = 0
        stops[r] = lut[r, np.int64((z >> np.uint64(12 * used)) & _LOW12)]
        used += 1
    return state


@njit(parallel=True, cache=True)
def _sessions(strips, lut, lines_flat, line_pay, vertical_pay, scatter_pay, slot_symbol, match,
              ladder, flags, n_sessions, n_spins, checkpoints, seed, max_hist, thresholds):
    n, L = strips.shape
    h = flags[0]
    nck = checkpoints.shape[0]
    cum = np.zeros((n_sessions, nck), np.int64)
    pay_sum = np.zeros(n_sessions, np.int64)
    pay_sq = np.zeros(n_sessions, np.int64)
    base_sum = np.zeros(n_sessions, np.int64)
    scatter_sum = np.zeros(n_sessions, np.int64)
    hits = np.zeros(n_sessions, np.int64)
    step_hist = np.zeros((n_sessions, max_hist + 1), np.int64)
    runaways = np.zeros(n_sessions, np.int64)
    nthr = thresholds.shape[0]
    tail = np.zeros((n_sessions, nthr), np.int64)   # spins paying at least each threshold
    for s in prange(n_sessions):
        state = _stream_state(seed, s)
        g = np.empty(n * h, np.int64)
        off = np.empty(n, np.int64)
        tmp = np.empty(h, np.int64)
        masks = np.empty(n, np.int64)
        stops = np.empty(n, np.int64)
        acc = 0
        acc_sq = 0
        acc_base = 0
        acc_scatter = 0
        n_hit = 0
        ci = 0
        for i in range(n_spins):
            state = _draw_stops(state, lut, stops)
            total, base, steps, ran_away, scat = _spin(stops, strips, lines_flat, line_pay,
                                                       vertical_pay, scatter_pay, slot_symbol,
                                                       match, ladder, flags, g, off, tmp, masks)
            acc += total
            acc_sq += total * total
            acc_base += base
            acc_scatter += scat
            if total > 0:
                n_hit += 1
            step_hist[s, min(steps, max_hist)] += 1
            runaways[s] += ran_away
            for j in range(nthr):
                if total >= thresholds[j]:
                    tail[s, j] += 1
            while ci < nck and checkpoints[ci] == i + 1:
                cum[s, ci] = acc
                ci += 1
        pay_sum[s] = acc
        pay_sq[s] = acc_sq
        base_sum[s] = acc_base
        scatter_sum[s] = acc_scatter
        hits[s] = n_hit
    return cum, pay_sum, pay_sq, base_sum, scatter_sum, hits, step_hist, runaways, tail


@njit(cache=True)
def spin_once(stops, strips, lines_flat, line_pay, vertical_pay, scatter_pay, slot_symbol, match,
              ladder, flags):
    """One spin from given stops, for testing the evaluator against the reference."""
    n = strips.shape[0]
    h = flags[0]
    g = np.empty(n * h, np.int64)
    off = np.empty(n, np.int64)
    tmp = np.empty(h, np.int64)
    masks = np.empty(n, np.int64)
    return _spin(stops, strips, lines_flat, line_pay, vertical_pay, scatter_pay, slot_symbol,
                 match, ladder, flags, g, off, tmp, masks)


@njit(parallel=True, cache=True)
def _every_stop(strips, lines_flat, line_pay, vertical_pay, scatter_pay, slot_symbol, match,
                ladder, flags):
    n, L = strips.shape
    h = flags[0]
    block = L ** (n - 1)
    total = np.empty(L * block, np.int64)
    steps = np.empty(L * block, np.int64)
    for first in prange(L):
        g = np.empty(n * h, np.int64)
        off = np.empty(n, np.int64)
        tmp = np.empty(h, np.int64)
        masks = np.empty(n, np.int64)
        stops = np.empty(n, np.int64)
        for j in range(block):
            stops[0] = first
            rest = j
            for r in range(n - 1, 0, -1):
                stops[r] = rest % L
                rest //= L
            pay, base, d, ran_away, scat = _spin(stops, strips, lines_flat, line_pay,
                                                 vertical_pay, scatter_pay, slot_symbol, match,
                                                 ladder, flags, g, off, tmp, masks)
            total[first * block + j] = pay
            steps[first * block + j] = d
    return total, steps


def evaluate_every_stop(game: Game) -> tuple[np.ndarray, np.ndarray]:
    """This evaluator's total pay and cascade steps for every stop vector, reel 1's stop the
    most significant base-L digit. For testing it against the exact enumeration in full."""
    return _every_stop(*spin_args(kernel_arrays(game)))


# ----------------------------------------------------------------------- driver
def kernel_arrays(game: Game) -> dict[str, np.ndarray]:
    """The game's arrays plus what this evaluator needs: flattened paylines, the slot each paying
    symbol occupies in a line's bitmask, each symbol's match set, and the sampling tables."""
    a = game.arrays()
    h = game.rows
    lines_flat = np.array([[r * h + row for r, row in enumerate(line)] for line in game.lines],
                          dtype=np.int64)
    slot_symbol = a["paying"]
    if slot_symbol.shape[0] > 62:
        raise ValueError("at most 62 paying symbols fit the line bitmask")
    match = np.zeros(len(game.symbols), dtype=np.int64)
    for i, s in enumerate(slot_symbol):
        match[s] = 1 << i
    wild = int(a["flags"][1])
    if wild >= 0:
        match[wild] = (1 << slot_symbol.shape[0]) - 1
    return {**a, "lines_flat": lines_flat, "slot_symbol": slot_symbol, "match": match,
            "lut": lookup_tables(game.weights)}


def spin_args(a: dict) -> tuple:
    """The evaluator's game arguments, in order, from ``kernel_arrays``."""
    return (a["strips"], a["lines_flat"], a["line_pay"], a["vertical_pay"], a["scatter_pay"],
            a["slot_symbol"], a["match"], a["ladder"], a["flags"])


def checkpoints_geometric(limit: int, ratio: float = 1.5) -> np.ndarray:
    """a_n = round(ratio ** n), n = 0, 1, 2, ... up to ``limit``, duplicates removed."""
    out, n = [], 0
    while round(ratio ** n) <= limit:
        out.append(round(ratio ** n))
        n += 1
    return np.unique(np.array(out, dtype=np.int64))


@dataclass(frozen=True)
class SimResult:
    """Pooled Monte Carlo output. Pays are in units; ``cum[s, i]`` is session s's total pay after
    ``checkpoints[i]`` spins."""

    stake_units: int
    sessions: int
    spins_per_session: int
    checkpoints: np.ndarray
    cum: np.ndarray
    pay_sum: np.ndarray
    pay_sq: np.ndarray
    base_sum: np.ndarray
    scatter_sum: np.ndarray
    hits: np.ndarray
    step_hist: np.ndarray
    runaways: int
    seconds: float
    threads: int
    thresholds: np.ndarray      # in units
    tail: np.ndarray            # per session: spins paying at least each threshold

    @property
    def spins(self) -> int:
        return self.sessions * self.spins_per_session

    @property
    def spins_per_second(self) -> float:
        return self.spins / self.seconds

    def _moments(self) -> tuple[float, float]:
        n = self.spins
        return int(self.pay_sum.sum()) / n, int(self.pay_sq.sum()) / n

    @property
    def rtp(self) -> float:
        return self._moments()[0] / self.stake_units

    @property
    def volatility(self) -> float:
        m1, m2 = self._moments()
        return sqrt(max(m2 - m1 * m1, 0.0)) / self.stake_units

    @property
    def rtp_se(self) -> float:
        return self.volatility / sqrt(self.spins)

    @property
    def base_rtp(self) -> float:
        return int(self.base_sum.sum()) / self.spins / self.stake_units

    @property
    def scatter_rtp(self) -> float:
        return int(self.scatter_sum.sum()) / self.spins / self.stake_units

    @property
    def hit_frequency(self) -> float:
        return int(self.hits.sum()) / self.spins

    @property
    def steps_distribution(self) -> np.ndarray:
        return self.step_hist.sum(axis=0) / self.spins

    @property
    def expected_steps(self) -> float:
        p = self.steps_distribution
        return float((np.arange(p.size) * p).sum())

    @property
    def expected_steps_se(self) -> float:
        p = self.steps_distribution
        d = np.arange(p.size)
        var = float((d * d * p).sum()) - self.expected_steps ** 2
        return sqrt(var / self.spins)

    def p_win_at_least(self) -> np.ndarray:
        """Observed frequency of spins paying at least each threshold."""
        return self.tail.sum(axis=0) / self.spins

    def session_rtp(self) -> np.ndarray:
        """RTP of every session after each checkpoint: shape (sessions, checkpoints)."""
        return self.cum / (self.checkpoints[None, :] * self.stake_units)


def simulate(game: Game, sessions: int, spins_per_session: int, *, seed: int = 2026,
             checkpoints: np.ndarray | None = None, threads: int | None = None,
             big_wins: tuple[int, ...] = ()) -> SimResult:
    """Run ``sessions`` independent sessions of ``spins_per_session`` spins each, in parallel.

    ``big_wins`` are multiples of the stake; the result counts spins paying at least each.
    """
    a = kernel_arrays(game)
    ck = (checkpoints_geometric(spins_per_session) if checkpoints is None
          else np.asarray(checkpoints, dtype=np.int64))
    thresholds = np.array([m * game.stake_units for m in big_wins], dtype=np.int64)
    previous = numba.get_num_threads()
    if threads is not None:
        numba.set_num_threads(threads)
    try:
        t0 = time.perf_counter()
        out = _sessions(a["strips"], a["lut"], a["lines_flat"], a["line_pay"], a["vertical_pay"],
                        a["scatter_pay"], a["slot_symbol"], a["match"], a["ladder"], a["flags"],
                        sessions, spins_per_session, ck, seed, game.max_steps, thresholds)
        seconds = time.perf_counter() - t0
        used = numba.get_num_threads()
    finally:
        numba.set_num_threads(previous)
    cum, pay_sum, pay_sq, base_sum, scatter_sum, hits, step_hist, runaways, tail = out
    return SimResult(game.stake_units, sessions, spins_per_session, ck, cum, pay_sum, pay_sq,
                     base_sum, scatter_sum, hits, step_hist, int(runaways.sum()), seconds, used,
                     thresholds, tail)


def simulate_python(game: Game, spins: int, seed: int = 2026, stream: int = 0) -> dict:
    """Pure-Python twin of one session: same random stream, same stops, reference evaluator.

    Bit-for-bit comparable with ``simulate(game, 1, spins, seed=seed)``, and the baseline for
    the efficiency comparison.
    """
    a = game.arrays()
    lut = lookup_tables(game.weights)
    state = stream_state_python(seed, stream)
    n = game.reels
    total = total_sq = base_sum = scatter_sum = hits = 0
    step_counts: dict[int, int] = {}
    t0 = time.perf_counter()
    for _ in range(spins):
        state, z = splitmix64_python(state)
        used = 0
        stops = []
        for r in range(n):
            if used == 5:
                state, z = splitmix64_python(state)
                used = 0
            stops.append(int(lut[r, (z >> (12 * used)) & 0xFFF]))
            used += 1
        pay, base, steps, _, scatter = reference_play(tuple(stops), game, a)
        total += pay
        total_sq += pay * pay
        base_sum += base
        scatter_sum += scatter
        hits += pay > 0
        step_counts[steps] = step_counts.get(steps, 0) + 1
    seconds = time.perf_counter() - t0
    return {"pay_sum": total, "pay_sq": total_sq, "base_sum": base_sum,
            "scatter_sum": scatter_sum, "hits": hits, "steps": step_counts, "seconds": seconds,
            "spins": spins}
