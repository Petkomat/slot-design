"""Fit reel strips and stop weights to a target RTP and volatility.

The inputs are fixed: the eight-symbol paytable, 5 reels of 32 positions, wilds, scatter,
cascades, an RTP of 0.96 and a volatility, the standard deviation of one spin's payout per unit
staked. What is fitted is what sits on the reels and how likely each stop is. The procedure is
the one a game mathematician runs by hand, made systematic:

1. **Strips.** A strip is a composition (how many of each symbol a reel carries) laid out by an
   arrangement rule. Three knobs span the designs: the number of high symbols per reel, the
   height of their stacks, and the number of wilds on reels 2-4 (reels 1 and 5 carry none, as is
   common). Every reel carries one scatter; low symbols fill the rest. Blocks are reshuffled with
   a fixed seed until no two blocks of the same symbol touch (rejection sampling, so every valid
   layout is equally likely), and a stack is exactly as tall as asked. Each candidate is
   enumerated exactly, once.

2. **Weights.** Both E[X] and E[X^2] are linear in each reel's weights (multilinear overall), so
   for a candidate strip set the weights that hit both targets come from a few Gauss-Newton
   steps. Among the many solutions, each step takes the one with the smallest change in log
   weight from uniform, within bounds (a stop's weight stays between a quarter and four times
   uniform). A target the bounds cannot reach is reported as infeasible for that candidate.

3. **Choice.** Of the candidates that reach a target, the one whose weights stay closest to
   uniform wins: the strip carries the design, the weights only fine-tune it.

4. **Integers.** The weights are rounded to integers summing to 4096 per reel and polished
   with single-unit moves. Moving one unit from stop i to stop j on reel r changes E[X] by
   exactly ``(G1_r[j] - G1_r[i]) / 4096`` and E[X^2] likewise, with G the conditional moments
   given the stop, so every candidate move is priced exactly. The result is verified with exact
   rational arithmetic in ``exact.py``; the float search only proposes.

    python -m slotmath.fit            # writes slotmath/designs.json (several minutes)
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .exact import enumerate_spins, exact_result
from .game import (
    DESIGNS_FILE,
    HIGH,
    LOW,
    WEIGHT_TOTAL,
    eight_symbol_game,
    uniform_weights,
)

LENGTH = 32
REELS = 5
TARGET_RTP = 0.96
WEIGHT_RATIO = 4.0                    # a stop's weight stays within 1/4 .. 4 times uniform
WEIGHT_BOUNDS = (int(WEIGHT_TOTAL / LENGTH / WEIGHT_RATIO),
                 int(WEIGHT_TOTAL / LENGTH * WEIGHT_RATIO))   # (32, 512) for L = 32
RTP_TOLERANCE = 1e-7                  # after integer polishing
VOLATILITY_TOLERANCE = 1e-3           # relative, after integer polishing

HIGH_RATIOS = (1, 2, 3)               # H1 : H2 : H3, rarest first
LOW_RATIOS = (2, 2.5, 3, 3.5, 4)      # L1 .. L5


# ------------------------------------------------------------------------ strips
@dataclass(frozen=True)
class StripParams:
    """One point of the strip design space."""

    high: int        # high symbols per reel
    stack: int       # height of a high-symbol stack
    wilds: int       # wilds on each of reels 2-4
    seed: int = 1    # arrangement seed

    def label(self) -> str:
        return f"high={self.high} stack={self.stack} wilds={self.wilds} seed={self.seed}"


def _split(total: int, ratios: tuple[float, ...]) -> list[int]:
    """
    Absolute counts of symbols from proportions ``ratios``,
    at least 1 each, summing to ``total``.
    """
    k = len(ratios)
    # reserve k places (+1 at the end) and round down the rest
    raw = np.array(ratios, dtype=float) / sum(ratios) * (total - k)
    c = np.floor(raw).astype(int)
    # round up the largest remainders
    short = total - k - int(c.sum())
    c[np.argsort(raw - c)[k - short:]] += 1
    # add one to all
    return [int(v) + 1 for v in c]


def composition(params: StripParams, reel: int) -> dict[str, int]:
    """How many of each symbol reel ``reel`` (0-based) carries."""
    wilds = params.wilds if 0 < reel < REELS - 1 else 0
    counts: dict[str, int] = dict(zip(HIGH, _split(params.high, HIGH_RATIOS)))
    counts.update(zip(LOW, _split(LENGTH - params.high - wilds - 1, LOW_RATIOS)))
    counts["W"] = wilds
    counts["S"] = 1
    return counts


def arrange(counts: dict[str, int], stack: int, seed: int) -> tuple[str, ...]:
    """Lay a composition out on a circular strip: high symbols in stacks of ``stack``, every
    other symbol singly, no two blocks of one symbol touching."""
    blocks: list[tuple[str, ...]] = []
    for s, c in counts.items():
        size = stack if s in HIGH else 1
        while c > 0:
            b = min(size, c)
            blocks.append((s,) * b)
            c -= b
    # Feasibility: around a circle, a symbol can fill at most every other block.
    m = len(blocks)
    most = max(Counter(b[0] for b in blocks).values())
    if 2 * most > m:
        raise ValueError(f"{most} blocks of one symbol cannot be kept apart among {m}")

    # Rejection sampling: shuffle until no two blocks of one symbol touch on the cycle.
    # A uniform shuffle over valid layouts is ... hard, so we sample all the layouts instead.
    # The number of shuffles is geometric with mean M = 1 / alpha, alpha
    # the share of shuffles that are valid (alfa = 0.0083 in the forst case in our grid)
    rng = random.Random(seed)
    while True:
        rng.shuffle(blocks)
        if all(blocks[i][0] != blocks[(i + 1) % m][0] for i in range(m)):
            break
    strip = tuple(s for b in blocks for s in b)
    assert len(strip) == LENGTH
    return strip


def strips_for(params: StripParams) -> tuple[tuple[str, ...], ...]:
    return tuple(arrange(composition(params, r), params.stack, params.seed * 10 + r)
                 for r in range(REELS))


# ------------------------------------------------------------- moments in weights
class Moments:
    """E[X] and E[X^2] of one spin's payout (in units) as functions of the reels' stop
    probabilities, from an exact enumeration."""

    def __init__(self, pay_total: np.ndarray, length: int, reels: int):
        x = pay_total.astype(np.float64)
        self.tensors = (x, x * x)
        self.length, self.reels = length, reels

    def evaluate(self, p: np.ndarray) -> list[tuple[float, list[np.ndarray]]]:
        """For each moment: its value and ``G[r][t]``, the moment given reel r stops at t (the
        other reels at ``p``). The moment is linear in reel r's probabilities: sum_t p[r,t] G[r][t].
        """
        L, n = self.length, self.reels
        shape = (L,) * (n - 1)
        out = []
        for x in self.tensors:
            a = (p[0] @ x.reshape(L, -1)).reshape(shape)          # reel 1 contracted
            b = (x.reshape(-1, L) @ p[n - 1]).reshape(shape)      # reel n contracted
            g = [b]
            for r in range(n - 2, 0, -1):
                g[0] = g[0] @ p[r]
            for r in range(1, n):
                t = a
                for s in range(n - 1, 0, -1):     # axis s - 1 of ``a`` holds reel s
                    if s != r:
                        t = np.tensordot(t, p[s], axes=([s - 1], [0]))
                g.append(t)
            out.append((float(g[0] @ p[0]), g))
        return out


def _softmax_rows(theta: np.ndarray) -> np.ndarray:
    e = np.exp(theta - theta.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class WeightSolution:
    p: np.ndarray              # stop probabilities, one row per reel
    rtp: float
    volatility: float
    iterations: int
    feasible: bool

    def is_better_than(self, other: 'WeightSolution') -> bool:
        """Self bettern then other:

        - if self not feasable, it cannot be better
        - if self is feasable, but other is not, it is always better
        - if both are feasable, the one with the lower distortion wins

        Args:
            other (WeightSolution): _description_

        Returns:
            bool: _description_
        """
        if not self.feasible:
            return False
        elif not other.feasible:
            return True
        else:
            return self.distortion < other.distortion

    @property
    def distortion(self) -> float:
        """RMS of log(weight / uniform weight): read as sqrt(mean(log(p) - log(1/L))).
          0 for uniform weights."""
        return float(np.sqrt(np.mean(np.log(self.p * self.p.shape[1]) ** 2)))


def _bounded_logs(theta: np.ndarray, ratio: float) -> np.ndarray:
    """Logs of L * p with p = softmax(theta), each held within [1/ratio, ratio]."""
    L = theta.shape[1]
    for _ in range(50):
        theta = np.log(_softmax_rows(theta) * L)
        if np.abs(theta).max() <= np.log(ratio) + 1e-12:
            break
        theta = np.clip(theta, -np.log(ratio), np.log(ratio))
    return theta


def solve_weights(mom: Moments, stake: int, rtp: float, volatility: float,
                  ratio: float = WEIGHT_RATIO, max_iter: int = 40) -> WeightSolution:
    """Stop probabilities with RTP ``rtp`` and volatility ``volatility``, as close to uniform as
    the Gauss-Newton steps find, each within ``ratio`` times uniform either way. The unknowns
    are ``theta = log(L * p)``, zero for uniform weights."""
    n, L = mom.reels, mom.length
    theta = np.zeros((n, L))
    for it in range(max_iter + 1):
        p = _softmax_rows(theta)
        (m1, g1), (m2, g2) = mom.evaluate(p)
        r1 = m1 / stake
        var = m2 / stake**2 - r1 * r1
        c = np.array([r1 - rtp, var - volatility**2])
        if abs(c[0]) < 1e-11 and abs(c[1]) < 1e-9 * volatility**2:
            return WeightSolution(p, r1, float(np.sqrt(var)), it, True)
        if it == max_iter:
            break
        # d/dtheta of sum_t p_t G_t under a softmax: p_t (G_t - m)
        j1 = np.stack([p[r] * (g1[r] - m1) for r in range(n)]) / stake
        j2 = np.stack([p[r] * (g2[r] - m2) for r in range(n)]) / stake**2 - 2 * r1 * j1
        jac = np.stack([j1.ravel(), j2.ravel()])
        lam = np.linalg.solve(jac @ jac.T, jac @ theta.ravel() - c)
        step = (jac.T @ lam).reshape(n, L) - theta   # to the minimum-norm linearised solution
        biggest = np.abs(step).max()
        if biggest > 0.5:
            step *= 0.5 / biggest
        theta = _bounded_logs(theta + step, ratio)
    return WeightSolution(p, r1, float(np.sqrt(max(var, 0.0))), it, False)


def round_weights(p: np.ndarray, bounds: tuple[int, int] = WEIGHT_BOUNDS) -> np.ndarray:
    """Integer weights summing to 4096 per reel, by largest remainder, then nudged one unit at a
    time into ``bounds``."""
    raw = p * WEIGHT_TOTAL
    w = np.floor(raw).astype(np.int64)
    for r in range(w.shape[0]):
        short = WEIGHT_TOTAL - int(w[r].sum())
        w[r, np.argsort(raw[r] - w[r])[w.shape[1] - short:]] += 1
        while w[r].min() < bounds[0]:
            w[r, np.argmin(w[r])] += 1
            w[r, np.argmax(w[r])] -= 1
        while w[r].max() > bounds[1]:
            w[r, np.argmax(w[r])] -= 1
            w[r, np.argmin(w[r])] += 1
    return w


def polish(mom: Moments, w: np.ndarray, stake: int, rtp: float, volatility: float,
           bounds: tuple[int, int] = WEIGHT_BOUNDS, max_moves: int = 400) -> tuple[np.ndarray, int]:
    """Greedy single-unit weight moves, each priced exactly, until none brings the pair
    (RTP, volatility) closer to the targets. RTP is weighted far above volatility."""
    w = w.copy()
    scale_r, scale_v = 1e-9, 1e-5 * volatility

    def error(m1, m2):
        r = m1 / stake
        v = np.sqrt(np.maximum(m2 / stake**2 - r * r, 0.0))
        return ((r - rtp) / scale_r) ** 2 + ((v - volatility) / scale_v) ** 2

    moves = 0
    while moves < max_moves:
        (m1, g1), (m2, g2) = mom.evaluate(w / WEIGHT_TOTAL)
        now = error(m1, m2)
        best = (now, None)
        for r in range(w.shape[0]):
            d1 = (g1[r][None, :] - g1[r][:, None]) / WEIGHT_TOTAL   # [i, j]: one unit i -> j
            d2 = (g2[r][None, :] - g2[r][:, None]) / WEIGHT_TOTAL
            e = error(m1 + d1, m2 + d2)
            e[w[r] - 1 < bounds[0], :] = np.inf
            e[:, w[r] + 1 > bounds[1]] = np.inf
            np.fill_diagonal(e, np.inf)
            i, j = np.unravel_index(np.argmin(e), e.shape)
            if e[i, j] < best[0]:
                best = (e[i, j], (r, i, j))
        if best[1] is None:
            break
        r, i, j = best[1]
        w[r, i] -= 1
        w[r, j] += 1
        moves += 1
    return w, moves


# ------------------------------------------------------------------------ search
GRID_HIGH = (3, 5, 7, 9, 11, 13, 15, 17)
GRID_STACK = (1, 2, 3, 4, 6)
GRID_WILDS = (1, 2, 3)


DEFAULT_GRID = tuple(itertools.starmap(StripParams,
                                       itertools.product(GRID_HIGH, GRID_STACK, GRID_WILDS)))


def finish(strips, solution: WeightSolution, rtp: float, volatility: float) -> dict:
    """Round and polish one design, then verify it exactly."""
    game = eight_symbol_game(strips, uniform_weights(LENGTH, REELS))
    spins = enumerate_spins(game)
    mom = Moments(spins.pay_total, LENGTH, REELS)
    w, moves = polish(mom, round_weights(solution.p), game.stake_units, rtp, volatility)
    ex = exact_result(spins, w)
    vol = ex.volatility
    ok = (abs(float(ex.rtp) - rtp) <= RTP_TOLERANCE
          and abs(vol - volatility) <= VOLATILITY_TOLERANCE * volatility)
    return {"weights": w.tolist(), "polish_moves": moves, "accepted": ok,
            "rtp_exact": f"{ex.rtp.numerator}/{ex.rtp.denominator}", "rtp": float(ex.rtp),
            "volatility": vol, "hit_frequency": float(ex.hit_frequency),
            "max_win": float(ex.max_win)}


def fit_designs(targets: list[float], grid: tuple[StripParams, ...], rtp: float = TARGET_RTP,
                log=print) -> dict:
    """Scan the strip grid, solve weights for every target on every candidate, keep the least
    distorted feasible candidate per target, then round, polish and verify the winners."""
    t0 = time.perf_counter()
    seen: set = set()
    scan = []
    dummy_params = StripParams(0, 0, 0, 0)
    dummy_solution = WeightSolution(np.array([[0.0]]), 0.0, 0.0, 0, False)
    best = dict.fromkeys(targets, (dummy_params, dummy_solution))
    for params in grid:
        strips = strips_for(params)
        if strips in seen:
            continue
        seen.add(strips)
        game = eight_symbol_game(strips, uniform_weights(LENGTH, REELS))
        spins = enumerate_spins(game)
        mom = Moments(spins.pay_total, LENGTH, REELS)
        (m1, _), (m2, _) = mom.evaluate(np.full((REELS, LENGTH), 1 / LENGTH))
        u_rtp = m1 / game.stake_units
        u_vol = float(np.sqrt(m2 / game.stake_units**2 - u_rtp**2))
        row = {"params": params.__dict__, "uniform_rtp": u_rtp, "uniform_volatility": u_vol,
               "distortion": {}}
        for v in targets:
            sol = solve_weights(mom, game.stake_units, rtp, v)
            row["distortion"][str(v)] = sol.distortion if sol.feasible else None
            if sol.is_better_than(best[v][1]):
                best[v] = (params, sol)
        scan.append(row)
        reach = [v for v in targets if row["distortion"][str(v)] is not None]
        log(f"{len(scan):3d} {params.label():34s} uniform RTP {u_rtp:.4f} vol {u_vol:6.3f}"
            f"  reaches {reach}")
    scan_seconds = time.perf_counter() - t0
    games = {}
    for k, v in enumerate(targets, start=1):
        if not best[v][1].feasible:
            log(f"volatility {v}: no candidate reaches it")
            continue
        params, sol = best[v]
        t1 = time.perf_counter()
        strips = strips_for(params)
        done = finish(strips, sol, rtp, v)
        games[f"v{k}"] = {"target_volatility": v, "params": params.__dict__,
                          "strips": [list(s) for s in strips],
                          "solver": {"iterations": sol.iterations, "distortion": sol.distortion},
                          **done, "finish_seconds": time.perf_counter() - t1}
        log(f"volatility {v}: {params.label()}, distortion {sol.distortion:.3f}, "
            f"{done['polish_moves']} polish moves, RTP {done['rtp']:.10f}, "
            f"vol {done['volatility']:.6f}, accepted {done['accepted']}")
    return {"inputs": {"rtp": rtp, "volatility_targets": targets, "length": LENGTH,
                       "reels": REELS, "weight_bounds": list(WEIGHT_BOUNDS),
                       "grid": {"high": list(GRID_HIGH), "stack": list(GRID_STACK),
                                "wilds": list(GRID_WILDS)}},
            "games": games, "scan": scan,
            "seconds": {"scan": scan_seconds, "total": time.perf_counter() - t0}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", type=float, nargs="+", default=list(VOLATILITY_TARGETS))
    parser.add_argument("--main", type=int, default=3, help="1-based index of the main game")
    parser.add_argument("--out", default=str(DESIGNS_FILE))
    args = parser.parse_args()
    designs = fit_designs(args.targets, DEFAULT_GRID)
    designs["main"] = f"v{args.main}"
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(designs, f, indent=1)
        f.write("\n")
    print(f"wrote {args.out} in {designs['seconds']['total']:.0f}s")


# Roughly geometric, inside the range a wider scan found reachable at RTP 0.96 with this paytable,
# grid and weight bounds: 2.5 to 8 all are, 1.5, 2, 10 and 12 are not
# (python -m slotmath.fit --targets 1.5 2 2.5 3 4 5 6 7 8 10 12 --out scan.json).
VOLATILITY_TARGETS = (2.5, 3.5, 4.5, 6.0, 8.0)

if __name__ == "__main__":
    main()
