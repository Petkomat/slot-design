"""The rules, written to be read. Pure Python, no speed tricks.

This module is the specification. The compiled evaluators in ``exact.py`` and ``simulate.py``
are written independently of it and of each other, and the tests require all three to agree on
every spin they are given.

Rules, per window:

* **Lines.** For each payline and each paying symbol ``s``, the run length ``k`` is the number
  of leading cells, from reel 1, that show ``s`` or (with wilds on) the wild. The line pays
  ``line_pay[s][k]``. Only the best symbol on a line pays: the highest pay, and on a tie the
  longer run. Its ``k`` leading cells are the line's winning cells.
* **Verticals** (optional). A reel whose visible column is one paying symbol pays
  ``vertical_pay[s]``; wilds do not substitute. All its cells are winning cells.
* **Cascades** (optional). Winning cells clear, survivors drop, and each reel refills from its
  own strip: clearing ``d`` cells moves the reel's offset back by ``d`` and brings in the ``d``
  symbols above the old window. Repeat until a window pays nothing. Cascade step ``j`` (the
  first win is step 0) is multiplied by ``ladder[min(j, len(ladder) - 1)]``.
* **Scatter** (optional). Scatters are never part of a line, so they never clear. Once the
  chain has ended, the visible scatters on the final window are counted and pay
  ``scatter_pay[count]`` (a multiple of the stake), unmultiplied, once per spin.
"""

from __future__ import annotations

from itertools import product

from .game import Game


def evaluate_window(grid: list[list[int]], game: Game, a: dict) -> tuple[int, set]:
    """Line and vertical pay of one window in units, and the set of winning cells."""
    n = game.reels
    wild = int(a["flags"][1])
    paying = [int(s) for s in a["paying"]]
    pay, cleared = 0, set()
    for line in game.lines:
        best_pay, best_k = 0, 0
        for s in paying:
            k = 0
            while k < n and grid[k][line[k]] in (s, wild):
                k += 1
            p = int(a["line_pay"][s, k])
            if p > 0 and (p, k) > (best_pay, best_k):
                best_pay, best_k = p, k
        if best_pay:
            pay += best_pay
            cleared.update((r, line[r]) for r in range(best_k))
    if game.verticals:
        for r in range(n):
            col = grid[r]
            if col[0] in paying and all(c == col[0] for c in col):
                pay += int(a["vertical_pay"][col[0]])
                cleared.update((r, k) for k in range(game.rows))
    return pay, cleared


def play(stops: tuple[int, ...], game: Game, a: dict | None = None
         ) -> tuple[int, int, int, bool, int]:
    """One spin from the given stops: (total pay, step-0 line pay, cascade steps, runaway?,
    scatter pay). The total includes the scatter pay.

    Cascade steps counts the windows with a line (or vertical) win: 0 when the lines pay
    nothing, 1 for a win that does not cascade into another, and so on.
    """
    a = a if a is not None else game.arrays()
    strips, L, h = a["strips"], game.length, game.rows
    offsets = list(stops)
    grid = [[int(strips[r, (offsets[r] + k) % L]) for k in range(h)] for r in range(game.reels)]
    total = base = steps = 0
    runaway = False
    while True:
        pay, cleared = evaluate_window(grid, game, a)
        if pay == 0:
            break
        pay *= game.ladder[min(steps, len(game.ladder) - 1)]
        total += pay
        if steps == 0:
            base = pay
        steps += 1
        if not game.cascades:
            break
        if steps >= game.max_steps:
            runaway = True
            break
        for r in range(game.reels):
            gone = [k for k in range(h) if (r, k) in cleared]
            if gone:
                keep = [grid[r][k] for k in range(h) if (r, k) not in cleared]
                offsets[r] = (offsets[r] - len(gone)) % L
                grid[r] = [int(strips[r, (offsets[r] + j) % L]) for j in range(len(gone))] + keep
    scatter = int(a["flags"][5])
    count = sum(c == scatter for col in grid for c in col)
    scatter_pay = int(a["scatter_pay"][count])
    return total + scatter_pay, base, steps, runaway, scatter_pay


def enumerate_exact(game: Game) -> dict:
    """Every stop vector, weighted: exact histograms of total pay, step-0 pay, cascade steps and
    scatter pay.

    Weights are Python integers, so nothing here is rounded. Only for small games: the loop
    visits all ``L ** n`` stop vectors in plain Python.
    """
    a = game.arrays()
    w = [list(map(int, row)) for row in a["weights"]]
    hist: dict[str, dict[int, int]] = {"pay": {}, "base": {}, "steps": {}, "scatter": {}}
    runaway = 0
    for stops in product(range(game.length), repeat=game.reels):
        weight = 1
        for r, t in enumerate(stops):
            weight *= w[r][t]
        total, base, steps, ran_away, scatter = play(stops, game, a)
        for key, value in (("pay", total), ("base", base), ("steps", steps),
                           ("scatter", scatter)):
            hist[key][value] = hist[key].get(value, 0) + weight
        runaway += ran_away
    return {**hist, "runaway": runaway}
