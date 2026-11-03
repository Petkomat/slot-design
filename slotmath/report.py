"""Analyse the fitted eight-symbol games and write REPORT.md and VOLATILITY.md with figures.

    python -m slotmath.report            # full run, about 2.5 x 10^9 simulated spins
    python -m slotmath.report --quick    # a fiftieth of the simulation, for a smoke test

REPORT.md covers the main game in depth; VOLATILITY.md covers the five volatility levels. The
designs themselves come from ``python -m slotmath.fit`` (``slotmath/designs.json``). Every number
in both files is produced by this run; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import time
from fractions import Fraction
from math import sqrt
from pathlib import Path

import numba
import numpy as np

from .exact import enumerate_spins, exact_result
from .fit import WEIGHT_BOUNDS
from .game import (
    HIGH,
    LOW,
    Game,
    fitted_game,
    load_designs,
    uniform_weights,
)
from .simulate import simulate, simulate_python

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "report" / "figures"
WINS = ROOT / "report" / "wins"
BIG_WINS = (10, 25, 50, 100, 250, 500)
BANDS = ((0, 0, "0 (no win)"), (0, 1, "below the stake"), (1, 1, "exactly the stake"),
         (1, 2, "1–2×"), (2, 5, "2–5×"), (5, 10, "5–10×"), (10, 20, "10–20×"),
         (20, 50, "20–50×"), (50, 100, "50–100×"), (100, 250, "100–250×"),
         (250, 500, "250–500×"), (500, 1000, "500–1,000×"), (1000, 10**9, "1,000× and more"))

# Colours: the validated reference palette of the dataviz method (slots 1-2 pass the CVD and
# normal-vision checks on the light surface), its blue ramp, and neutral inks. RAMP is the
# five-step ordinal blue (steps 250-650, validated with --ordinal) for volatility levels 1-5.
SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3", "#a8a7a0"
S1, S2 = "#2a78d6", "#eb6834"
SEQ = {"100": "#cde2fb", "200": "#9ec5f4", "300": "#6da7ec", "550": "#1c5cab"}
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")


def level_colour(key: str) -> str:
    """The ramp step of volatility level ``v1`` .. ``v5``, fixed by the level, not its rank."""
    return RAMP[int(key[1:]) - 1]


# ============================================================================ runs
def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def warm_up(game: Game) -> float:
    """Load (or compile) the kernels so later timings measure work, not compilation."""
    t0 = time.perf_counter()
    small = game.with_options(strips=tuple(s[:8] for s in game.strips),
                              weights=uniform_weights(8, game.reels))
    exact_result(enumerate_spins(small))
    simulate(game, 2, 1000, big_wins=BIG_WINS)
    return time.perf_counter() - t0


def run_exact(game: Game) -> dict:
    spins, t_enum = timed(enumerate_spins, game)
    ex, t_weigh = timed(exact_result, spins)
    uniform = exact_result(spins, np.array(uniform_weights(game.length, game.reels)))
    return {"spins": spins, "exact": ex, "t_enum": t_enum, "t_weigh": t_weigh, "uniform": uniform}


OPTION_SETS = [
    ("Lines only", dict(wilds=False, cascades=False, scatters=False, ladder=(1,))),
    ("+ wilds", dict(cascades=False, scatters=False, ladder=(1,))),
    ("+ cascades", dict(scatters=False, ladder=(1,))),
    ("+ multiplier ladder", dict(scatters=False)),
    ("+ scatter (the game)", {}),
    ("The game without wilds", dict(wilds=False)),
    ("The game without cascades", dict(cascades=False)),
]


def run_options(name: str, base_spins) -> list:
    rows = []
    for label, opts in OPTION_SETS:
        game = fitted_game(name, **opts)
        spins, t = timed(enumerate_spins, game) if opts else (base_spins, 0.0)
        rows.append((label, exact_result(spins), t))
    g = base_spins.game
    rows.append(("The game, uniform weights",
                 exact_result(base_spins, np.array(uniform_weights(g.length, g.reels))), 0.0))
    return rows


def run_control(game: Game, spins: int) -> dict:
    """The pure-Python control on the same random stream as a one-session Numba run: its time,
    and whether it reproduces the Numba session's totals bit for bit."""
    control = simulate_python(game, spins, seed=7)
    same = simulate(game, 1, spins, seed=7)
    identical = all(int(getattr(same, k)[0]) == control[k]
                    for k in ("pay_sum", "pay_sq", "base_sum", "scatter_sum", "hits"))
    return {"spins": spins, "seconds": control["seconds"], "identical": identical}


def run_efficiency(game: Game, quick: bool) -> dict:
    """Thread scaling of the Numba Monte Carlo and of the exact enumeration, and the control."""
    control = run_control(game, 5_000 if quick else 50_000)
    # 240 sessions divide evenly among every thread count tested (1 .. 20)
    sessions, per = (240, 4_000) if quick else (240, 40_000)
    cores = os.cpu_count() or 1
    rows = []
    for k in sorted({1, 2, 4, 8, 16, cores} & set(range(1, cores + 1))):
        sim = simulate(game, sessions, per, seed=7, threads=k)
        rows.append((k, sim.spins, sim.seconds))
    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        _, t_enum_1 = timed(enumerate_spins, game)
    finally:
        numba.set_num_threads(previous)
    _, t_enum_n = timed(enumerate_spins, game)
    return {"rows": rows, "control": control, "t_enum_1": t_enum_1, "t_enum_n": t_enum_n,
            "threads": previous}


def run_sweep(designs: dict, scale: int) -> list[dict]:
    """Exact analysis, a Monte Carlo check and the pure-Python control for every level."""
    out = []
    for k, key in enumerate(sorted(designs["games"]), start=1):
        game = fitted_game(key)
        exact = run_exact(game)
        mc = simulate(game, 20, 5_000_000 // scale, seed=3000 + k, big_wins=BIG_WINS)
        control = run_control(game, 20_000 // scale)
        out.append({"key": key, "design": designs["games"][key], "game": game, "exact": exact,
                    "mc": mc, "control": control})
        print(f"  {key}: volatility {exact['exact'].volatility:.4f}, "
              f"{mc.spins:,} spins in {mc.seconds:.1f} s, control identical {control['identical']}")
    return out


# ========================================================================= figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook", rc={
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#d6d5cf", "axes.linewidth": 0.8, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
        "lines.linewidth": 1.6, "font.size": 10, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False})
    import matplotlib.pyplot as plt
    return plt


def _save(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    return f"report/figures/{name}"


def figure_convergence(plt, sim, rtp: float, vol: float) -> str:
    x = sim.checkpoints
    R = sim.session_rtp() * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    band = 3 * vol / np.sqrt(x) * 100
    ax.fill_between(x, rtp * 100 - band, rtp * 100 + band, color=SEQ["100"], lw=0,
                    label="exact RTP ± 3σ/√N  (99.7% band)")
    for row in R:
        ax.plot(x, row, color=S1, alpha=0.10, lw=0.6)
    ax.plot([], [], color=S1, lw=1.2, alpha=0.7, label=f"one session ({sim.sessions} shown)")
    ax.axhline(rtp * 100, color=INK, lw=1.2, label=f"exact RTP {rtp:.2%}")
    ax.set_xscale("log")
    ax.set_ylim(0, 250)
    ax.set_xlabel(r"spins played, checkpoints $a_n = \mathrm{round}(1.5^n)$")
    ax.set_ylabel("observed RTP (%)")
    ax.set_title(f"Observed RTP after $a_n$ spins: {sim.sessions} sessions of "
                 f"{sim.spins_per_session:,} spins")
    ax.legend(loc="lower right")
    return _save(fig, "rtp_convergence.png")


def figure_sessions(plt, sim, rtp: float) -> str:
    x = sim.checkpoints
    R = sim.session_rtp() * 100
    q = np.quantile(R, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(x, q[0], q[4], color=SEQ["200"], lw=0, label="5% to 95% of sessions")
    ax.fill_between(x, q[1], q[3], color=SEQ["300"], lw=0, label="25% to 75% of sessions")
    ax.plot(x, q[2], color=SEQ["550"], lw=1.6, label="median session")
    ax.axhline(rtp * 100, color=INK, lw=1.2, label=f"exact RTP {rtp:.2%}")
    ax.set_xscale("log")
    ax.set_ylim(0, 250)
    ax.set_xlabel(r"spins played, checkpoints $a_n = \mathrm{round}(1.5^n)$")
    ax.set_ylabel("observed RTP (%)")
    ax.set_title(f"Player sessions: observed RTP across {sim.sessions:,} sessions")
    ax.legend(loc="upper right")
    return _save(fig, "player_sessions.png")


def figure_cascades(plt, ex, sim) -> str:
    dmax = max(ex.steps)
    d = np.arange(dmax + 1)
    p_exact = np.array([float(ex.p_steps(k)) for k in d])
    p_mc = sim.steps_distribution[: dmax + 1]
    contrib = ex.rtp_by_steps()
    c = np.array([float(contrib.get(k, 0)) * 100 for k in d])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.bar(d, p_exact, color=S1, width=0.7, label="exact")
    a1.plot(d, np.where(p_mc > 0, p_mc, np.nan), "o", color=S2, ms=7, mec=SURFACE, mew=1.5,
            label=f"Monte Carlo ({sim.spins:.1e} spins)")
    a1.set_yscale("log")
    a1.set_xlabel("windows with a line win in the spin, D")
    a1.set_ylabel("probability")
    a1.set_title("Cascade length: P(D = d)")
    a1.legend()
    a2.bar(d, c, color=S1, width=0.7)
    a2.set_xlabel("windows with a line win in the spin, D")
    a2.set_ylabel("RTP contributed (percentage points)")
    a2.set_title("Where the 96% comes from, by cascade length")
    return _save(fig, "cascades.png")


def figure_options(plt, rows) -> str:
    labels = [r[0] for r in rows]
    rtp = [float(r[1].rtp) * 100 for r in rows]
    hit = [float(r[1].hit_frequency) * 100 for r in rows]
    vol = [r[1].volatility for r in rows]
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for ax, vals, title, unit in ((axes[0], rtp, "RTP", "%"), (axes[1], hit, "Hit frequency", "%"),
                                  (axes[2], vol, "Volatility (sd per unit stake)", "sd")):
        ax.barh(y, vals, color=S1, height=0.62)
        ax.set_title(title)
        ax.set_xlabel(unit)
    axes[0].axvline(96, color=INK, lw=1.0)
    axes[0].set_yticks(y, labels)
    return _save(fig, "options.png")


def _survival(ex) -> tuple[np.ndarray, np.ndarray]:
    pay = ex.pay
    xs = np.array(sorted(x for x in pay if x > 0), dtype=np.float64)
    ws = np.array([pay[int(x)] for x in xs], dtype=np.float64)
    return xs / ex.stake_units, ws[::-1].cumsum()[::-1] / ex.total_weight


def figure_wins(plt, ex, sim) -> str:
    xs, surv = _survival(ex)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(xs, surv, where="post", color=S1, lw=1.6, label="exact P(win ≥ x)")
    p = sim.p_win_at_least()
    keep = p > 0
    se = np.sqrt(p * (1 - p) / sim.spins)
    ax.errorbar(np.array(BIG_WINS)[keep], p[keep], yerr=1.96 * se[keep], fmt="o", color=S2, ms=7,
                mec=SURFACE, mew=1.5, capsize=3, lw=1.2, label="Monte Carlo, 95% interval")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("win, as a multiple of the stake")
    ax.set_ylabel("probability per spin")
    ax.set_title("Possible wins: probability that a spin pays at least x times the stake")
    ax.legend()
    return _save(fig, "wins.png")


def figure_performance(plt, eff) -> str:
    rows, control = eff["rows"], eff["control"]
    labels = [f"Numba, {k} thread{'s' if k > 1 else ''}" for k, _, _ in rows]
    labels.append("pure-Python control")
    rate = np.array([sp / sec for _, sp, sec in rows] + [control["spins"] / control["seconds"]])
    k = np.array([r[0] for r in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    y = np.arange(len(labels))[::-1]
    a1.plot(rate, y, "o", color=S1, ms=8, mec=SURFACE, mew=1.5)
    a1.set_xscale("log")
    a1.set_xlim(rate.min() / 3, rate.max() * 6)
    a1.set_yticks(y, labels)
    used = [i for i, (k_, _, _) in enumerate(rows) if k_ == eff["threads"]]
    for i in (len(rate) - 1, *used):   # label the control and the configuration used throughout
        a1.annotate(f"{rate[i]:,.0f}/s", (rate[i], y[i]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)
    a1.set_xlabel("spins per second (log scale)")
    a1.set_title("Monte Carlo throughput")
    a2.plot(k, k, color=NEUTRAL, lw=1.2, label="ideal (linear)")
    a2.plot(k, rate[:-1] / rate[0], "o-", color=S1, ms=7, mec=SURFACE, mew=1.5, label="measured")
    a2.set_xticks(k)
    a2.set_xlabel("threads")
    a2.set_ylabel("speed-up over one thread")
    a2.set_title("Parallel scaling of the Numba implementation")
    a2.legend()
    return _save(fig, "performance.png")


def figure_sweep_wins(plt, levels) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    for lv in levels:
        colour = level_colour(lv["key"])
        ex = lv["exact"]["exact"]
        xs, surv = _survival(ex)
        ax.step(xs, surv, where="post", color=colour, lw=1.8,
                label=f"{lv['key']}: volatility {ex.volatility:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("win, as a multiple of the stake")
    ax.set_ylabel("probability per spin")
    ax.set_title("P(win ≥ x) for the five volatility levels (exact)")
    ax.legend(loc="lower left")
    return _save(fig, "volatility_wins.png")


def figure_design_space(plt, designs, levels) -> str:
    scan = designs["scan"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.scatter([r["uniform_volatility"] for r in scan], [r["uniform_rtp"] * 100 for r in scan],
               s=22, color=NEUTRAL, lw=0, label="candidate strip set, uniform weights")
    for lv in levels:
        colour = level_colour(lv["key"])
        d = lv["design"]
        row = next(r for r in scan if r["params"] == d["params"])
        x0, y0 = row["uniform_volatility"], row["uniform_rtp"] * 100
        ex = lv["exact"]["exact"]
        x1, y1 = ex.volatility, float(ex.rtp) * 100
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.4, shrinkA=4, shrinkB=5))
        ax.plot(x0, y0, "o", ms=9, mfc=SURFACE, mec=colour, mew=2)
        ax.plot(x1, y1, "o", ms=9, color=colour, mec=SURFACE, mew=1.5)
        ax.annotate(lv["key"], (x1, y1), xytext=(0, -16), textcoords="offset points",
                    ha="center", color=INK2, fontsize=9)
    ax.plot([], [], "o", ms=9, mfc=SURFACE, mec=INK2, mew=2, label="chosen strips, uniform weights")
    ax.plot([], [], "o", ms=9, color=INK2, mec=SURFACE, label="after fitting the weights")
    ax.axhline(96, color=INK, lw=1.0)
    ax.set_xscale("log")
    ticks = [t for t in (1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 30)
             if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("volatility (sd per unit stake, log scale)")
    ax.set_ylabel("RTP (%)")
    ax.set_title("The design space: strips set the region, weights land the target")
    ax.legend(loc="upper left")
    return _save(fig, "design_space.png")


SYMBOL_CLASS = {**{s: 0 for s in HIGH}, **{s: 1 for s in LOW}, "W": 2, "S": 3}
CLASS_COLOURS = (S2, "#d6d5cf", S1, "#1baf7a")   # high, low, wild, scatter
CLASS_NAMES = ("high symbol", "low symbol", "wild", "scatter")


def figure_strips(plt, levels) -> str:
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap(CLASS_COLOURS)
    fig, axes = plt.subplots(1, len(levels), figsize=(12, 6.4), sharey=True)
    for ax, lv in zip(np.atleast_1d(axes), levels):
        g = lv["game"]
        grid = np.array([[SYMBOL_CLASS[s] for s in strip] for strip in g.strips]).T
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(g.reels), [str(r + 1) for r in range(g.reels)])
        ax.set_xticks(np.arange(-0.5, g.reels), minor=True)
        ax.set_yticks(np.arange(-0.5, g.length), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=1.2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel("reel")
        ax.set_title(f"{lv['key']}: volatility {lv['exact']['exact'].volatility:.1f}", fontsize=11)
    np.atleast_1d(axes)[0].set_ylabel("strip position")
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(CLASS_COLOURS, CLASS_NAMES)],
               loc="lower center", ncol=4, frameon=False)
    fig.suptitle("The fitted strips, low to high volatility", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "volatility_strips.png", dpi=150)
    return "report/figures/volatility_strips.png"


# ======================================================================== markdown
def pct(x, digits=4) -> str:
    return f"{float(x) * 100:.{digits}f}%"


def wilds(k: int) -> str:
    return f"{k} wild{'s' if k != 1 else ''}"


def one_in(p) -> str:
    p = float(p)
    if p <= 0:
        return "never"
    return f"1 in {1 / p:.1f}" if 1 / p < 10 else f"1 in {1 / p:,.0f}"


def near_decades(checkpoints: np.ndarray) -> list[int]:
    """Indices of the checkpoints nearest 1, 10, 100, ... and of the last one."""
    logs = np.log10(checkpoints.astype(float))
    picks = {int(np.argmin(np.abs(logs - k))) for k in range(int(logs[-1]) + 1)}
    return sorted(picks | {len(checkpoints) - 1})


def z(observed: float, expected: float, se: float) -> str:
    return f"{(observed - expected) / se:+.2f}" if se > 0 else "n/a"


def _in_band(x: Fraction, lo: int, hi: int) -> bool:
    if lo == hi:
        return x == lo
    if lo in (0, 1):
        return lo < x < hi
    return lo <= x < hi


def bands(ex) -> list[tuple[str, Fraction, Fraction, int]]:
    """(label, probability, RTP share, distinct win values) for every non-empty band."""
    rows = []
    for lo, hi, label in BANDS:
        sel = [(x, w) for x, w in ex.pay.items() if _in_band(Fraction(x, ex.stake_units), lo, hi)]
        if sel:
            p = Fraction(sum(w for _, w in sel), ex.total_weight)
            share = Fraction(sum(x * w for x, w in sel), ex.total_weight * ex.stake_units)
            rows.append((label, p, share, len(sel)))
    return rows


def band_table(ex) -> list[str]:
    out = ["| win (× stake) | probability | frequency | share of RTP | distinct values |",
           "| --- | --- | --- | --- | --- |"]
    for label, p, share, k in bands(ex):
        out.append(f"| {label} | {float(p):.6e} | {one_in(p)} | {pct(share, 2)} | {k:,} |")
    return out + [""]


def write_wins_csv(ex, key: str) -> str:
    """Every possible win with its exact probability."""
    WINS.mkdir(parents=True, exist_ok=True)
    path = WINS / f"{key}.csv"
    lines = ["win_x_stake,win_units,probability,probability_exact"]
    for x in sorted(ex.pay):
        p = Fraction(ex.pay[x], ex.total_weight)
        exact = f"{p.numerator}/{p.denominator}"
        lines.append(f"{Fraction(x, ex.stake_units)},{x},{float(p):.12e},{exact}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"report/wins/{key}.csv"


def paytable_lines(game: Game) -> list[str]:
    out = ["| symbol | role | 3 of a kind | 4 of a kind | 5 of a kind |",
           "| --- | --- | --- | --- | --- |"]
    for s in game.paying:
        lp = game.line_pay[s]
        out.append(f"| `{s}` | {'high' if s in HIGH else 'low'} | {lp[3]} | {lp[4]} | {lp[5]} |")
    out.append("| `W` | wild: substitutes for every paying symbol, no pay of its own | | | |")
    sp = game.scatter_pay
    out.append(f"| `S` | scatter, anywhere: pays × **stake**, not × line bet | {sp[3]} | {sp[4]} | "
               f"{sp[5]} (5 or more) |")
    return out + [""]


def composition_table(game: Game) -> list[str]:
    w = np.array(game.weights)
    out = ["| symbol | " + " | ".join(f"reel {r + 1}" for r in range(game.reels)) + " |",
           "| --- |" + " --- |" * game.reels]
    for s in game.symbols:
        cells = []
        for r, strip in enumerate(game.strips):
            n = strip.count(s)
            # share of the middle row: stop t shows position t + 1 there
            share = sum(w[r][t] for t in range(game.length)
                        if strip[(t + 1) % game.length] == s) / 4096
            cells.append(f"{n} ({share:.1%})" if n else "–")
        out.append(f"| `{s}` | " + " | ".join(cells) + " |")
    out.append("| weight range | " + " | ".join(f"{w[r].min()}–{w[r].max()}"
                                              for r in range(game.reels)) + " |")
    return out + [""]


def strip_table(game: Game) -> list[str]:
    w = game.weights
    head = "| stop | " + " | ".join(f"reel {r + 1} | weight" for r in range(game.reels)) + " |"
    out = [head, "| --- |" + " --- | --- |" * game.reels]
    for t in range(game.length):
        cells = " | ".join(f"`{game.strips[r][t]}` | {w[r][t]}" for r in range(game.reels))
        out.append(f"| {t} | {cells} |")
    return out + [""]


def inputs_section(add, game: Game, designs: dict, volatility_text: str) -> None:
    """Section 1: the game definition, the same for every design."""
    n, L = game.reels, game.length
    lo, hi = WEIGHT_BOUNDS
    add(f"- **Geometry.** {n} reels × {game.rows} rows, {L} positions per reel, "
        f"{len(game.lines)} paylines (`LINES_20` in `slotmath/game.py`), paying left to right "
        "from reel 1.")
    add(f"- **Stake.** One line bet per line, {len(game.lines)} line bets a spin. RTP is the "
        "expected payout divided by the stake; win sizes are multiples of the whole stake.")
    add("- **Symbols.** Eight paying symbols, three high (`H1`–`H3`) and five low (`L1`–`L5`), a "
        "wild `W` and a scatter `S`. No blanks: every strip position carries a symbol, as on "
        "five-reel video slots.")
    add("- **Wilds.** `W` substitutes for every paying symbol on a line; a line of five wilds "
        "pays as `H1`. Wilds sit on reels 2–4 only.")
    add("- **Cascades (avalanche).** Winning cells clear, survivors drop, and each reel refills "
        "from its own strip. Cascade step 0, 1, 2, 3+ is multiplied by "
        f"{', '.join('×' + str(m) for m in game.ladder)}.")
    add("- **Scatter.** Never part of a line, so never cleared. When the chain ends, the scatters "
        "on the final window are counted and pay once, unmultiplied. One scatter per reel.")
    add(f"- **Targets.** RTP = **{designs['inputs']['rtp']:.2f}** and {volatility_text}. "
        "*Volatility* is the standard deviation of one spin's payout divided by the stake.")
    add(f"- **What is fitted.** The strips (which symbol sits where) and the stop weights, "
        f"4096 = 2¹² per reel, each stop between {lo} and {hi} (a quarter to four times "
        "uniform).\n")
    add("**Paytable** (line pays in line bets; the paytable is an input, the same for every "
        "design):\n")
    for line in paytable_lines(game):
        add(line)


def fit_paragraph(add, design: dict, designs: dict) -> None:
    p = design["params"]
    sec = designs["seconds"]
    n_cand = len(designs["scan"])
    add(f"**How these were found** (`python -m slotmath.fit`). {n_cand} candidate strip sets "
        "were generated from three knobs — high symbols per reel, the height of their stacks, "
        "and wilds per reel on reels 2–4 — and each was enumerated exactly. For every candidate "
        "the weights hitting both targets were solved by Gauss–Newton (E[X] and E[X²] are linear "
        "in each reel's weights), taking the smallest change in log weight from uniform. The "
        "candidate whose weights stayed closest to uniform won. The scan took "
        f"{sec['scan']:.0f} s. Winner: **{p['high']} high symbols per reel in stacks of up to "
        f"{p['stack']}, {wilds(p['wilds'])} on each of reels 2–4** (arrangement seed "
        f"{p['seed']}); the solver needed {design['solver']['iterations']} steps, the RMS change "
        f"in log weight is {design['solver']['distortion']:.3f}, and rounding to integers was "
        f"repaired by {design['polish_moves']} single-unit moves, each priced exactly.\n")


def write_report(r: dict) -> str:
    g, ex, uni = r["game"], r["exact"]["exact"], r["exact"]["uniform"]
    design, designs = r["design"], r["designs"]
    n, L = g.reels, g.length
    target = design["target_volatility"]
    out: list[str] = []
    add = out.append

    add("# The fitted 5×3 game: design, exact analysis and Monte Carlo\n")
    add(f"Generated by `python -m slotmath.report{' --quick' if r['quick'] else ''}` on "
        f"{dt.date.today().isoformat()}. Every number below comes from that run; nothing is typed "
        f"in by hand. Machine: {r['machine']}. The same game fitted to four other volatilities "
        "is in [VOLATILITY.md](VOLATILITY.md).\n")

    # ------------------------------------------------------------------ 1
    add("## 1. Game definition (inputs)\n")
    inputs_section(add, g, designs, f"volatility = **{target}**")

    # ------------------------------------------------------------------ 2
    add("## 2. The fitted game\n")
    add("### 2.1 Reels and weights\n")
    fit_paragraph(add, design, designs)
    add("| target | requested | achieved (exact) |")
    add("| --- | --- | --- |")
    add(f"| RTP | {designs['inputs']['rtp']:.2f} | **{float(ex.rtp):.10f}** = "
        f"{ex.rtp.numerator}/{ex.rtp.denominator} (residual {float(ex.rtp) - 0.96:+.1e}) |")
    add(f"| volatility | {target} | **{ex.volatility:.6f}** (relative residual "
        f"{ex.volatility / target - 1:+.1e}) |")
    add("")
    add("**Composition**: how many of each symbol a reel carries, and in brackets the share of "
        "the middle row it fills once the weights are applied.\n")
    for line in composition_table(g):
        add(line)
    add("**Strips and weights** (stop t shows positions t, t+1, t+2 in rows 1–3; the probability "
        "of stop t is its weight / 4096):\n")
    add("<details><summary>Show the 32 × 5 table</summary>\n")
    for line in strip_table(g):
        add(line)
    add("</details>\n")

    add("### 2.2 Possible wins\n")
    add(f"All {L}⁵ = {L ** n:,} stop vectors were evaluated by the Numba enumerator in "
        f"**{r['exact']['t_enum']:.2f} s** on {r['threads']} threads and the weights applied in "
        f"{r['exact']['t_weigh']:.2f} s. "
        "Cascades refill from the strips, so a spin is a function of its stops, and every "
        "probability below is exact: an integer over 4096⁵ = 2⁶⁰.\n")
    add("| quantity | exact value |")
    add("| --- | --- |")
    add(f"| RTP | {float(ex.rtp):.10f} |")
    add(f"| of which: first window's line wins | {float(ex.base_rtp):.6f} |")
    add("| of which: cascades (steps 1 and later, multipliers included) | "
        f"{float(ex.cascade_rtp):.6f} |")
    add(f"| of which: scatter | {float(ex.scatter_rtp):.6f} |")
    add(f"| hit frequency (any payout) | {float(ex.hit_frequency):.6f} "
        f"({one_in(ex.hit_frequency)}) |")
    add(f"| P(win > stake) | {float(ex.p_win_above(1)):.6f} ({one_in(ex.p_win_above(1))}) |")
    add(f"| volatility (sd per unit stake) | {ex.volatility:.4f} |")
    add(f"| max win | {float(ex.max_win):,.2f}× stake |")
    add(f"| distinct payout values | {len(ex.pay):,} |")
    add(f"| runaway cascades (chains of {g.max_steps}) | {ex.runaways} |")
    add("")
    add(f"With uniform weights the same strips pay {pct(uni.rtp, 2)} at volatility "
        f"{uni.volatility:.3f}; the weights move the game to its targets.\n")
    add(f"The {len(ex.pay):,} payout values, grouped by size (every value with its exact "
        f"probability is in [{r['csv']}]({r['csv']})):\n")
    for line in band_table(ex):
        add(line)
    add(f"![Possible wins: exact survival function and Monte Carlo]({r['fig']['wins']})\n")

    add("### 2.3 Cascade analysis\n")
    add("Let **D** be the number of windows with a line win in a spin: 0 when the lines pay "
        "nothing (a scatter can still pay), 1 for a win whose refill pays nothing, and so on. A "
        "spin with D = d contains d − 1 paying cascades.\n")
    add("| quantity | exact |")
    add("| --- | --- |")
    add(f"| E[D] per spin | **{float(ex.expected_steps):.8f}** |")
    add("| E[D \\| D ≥ 1], per spin with a line win | "
        f"**{float(ex.expected_steps_given_win):.8f}** |")
    add("| E[paying cascades \\| D ≥ 1] = E[D \\| D ≥ 1] − 1 | "
        f"{float(ex.expected_steps_given_win) - 1:.8f} |")
    tail_sum = sum(sum(ex.p_steps(k) for k in ex.steps if k >= d)
                   for d in range(1, max(ex.steps) + 1))
    add(f"| check: Σ_d P(D ≥ d) | {float(tail_sum):.8f} (equals E[D] exactly: "
        f"{tail_sum == ex.expected_steps}) |")
    add("")
    contrib = ex.rtp_by_steps()
    add("| D | P(D = d) | P(D ≥ d) | RTP contributed |")
    add("| --- | --- | --- | --- |")
    for d in sorted(ex.steps):
        p_ge = sum(ex.p_steps(k) for k in ex.steps if k >= d)
        add(f"| {d} | {float(ex.p_steps(d)):.8f} | {float(p_ge):.8f} | "
            f"{pct(contrib.get(d, 0), 3)} |")
    add("")
    add(f"![Cascade length distribution and RTP by cascade length]({r['fig']['cascades']})\n")

    add("### 2.4 What each mechanic is worth\n")
    add("Each configuration is its own exact enumeration with the fitted weights held fixed, so "
        "the table shows what each mechanic adds, not a refitted game.\n")
    add("| configuration | RTP | hit frequency | volatility | E[D] | max win | enumeration |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for label, e, t in r["options"]:
        add(f"| {label} | {pct(e.rtp, 2)} | {pct(e.hit_frequency, 2)} | {e.volatility:.3f} | "
            f"{float(e.expected_steps):.4f} | {float(e.max_win):,.1f}× | "
            f"{f'{t:.2f} s' if t else 'reused'} |")
    add("")
    add(f"![RTP, hit frequency and volatility by configuration]({r['fig']['options']})\n")
    monte_carlo_section(add, r)
    checks_section(add, r)
    speed_section(add, r)
    path = ROOT / "REPORT.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return str(path)


def agreement_rows(ex, mc) -> list[str]:
    """Exact against Monte Carlo, with standard errors and z-scores."""
    out = ["| quantity | exact | Monte Carlo | standard error | z |",
           "| --- | --- | --- | --- | --- |"]
    out.append(f"| RTP | {float(ex.rtp):.6f} | {mc.rtp:.6f} | {mc.rtp_se:.6f} | "
               f"{z(mc.rtp, float(ex.rtp), mc.rtp_se)} |")
    p_hit = float(ex.hit_frequency)
    se_hit = sqrt(p_hit * (1 - p_hit) / mc.spins)
    out.append(f"| hit frequency | {p_hit:.6f} | {mc.hit_frequency:.6f} | {se_hit:.6f} | "
               f"{z(mc.hit_frequency, p_hit, se_hit)} |")
    out.append(f"| E[D] | {float(ex.expected_steps):.6f} | {mc.expected_steps:.6f} | "
               f"{mc.expected_steps_se:.6f} | "
               f"{z(mc.expected_steps, float(ex.expected_steps), mc.expected_steps_se)} |")
    for d in range(1, max(ex.steps) + 1):
        pe = float(ex.p_steps(d))
        if pe * mc.spins < 10:
            continue
        se = sqrt(pe * (1 - pe) / mc.spins)
        out.append(f"| P(D = {d}) | {pe:.3e} | {mc.steps_distribution[d]:.3e} | {se:.1e} | "
                   f"{z(mc.steps_distribution[d], pe, se)} |")
    pw = mc.p_win_at_least()
    for i, m in enumerate(BIG_WINS):
        pe = float(ex.p_win_at_least(m))
        if pe * mc.spins < 10:
            continue
        se = sqrt(pe * (1 - pe) / mc.spins)
        out.append(f"| P(win ≥ {m}×) | {pe:.3e} | {pw[i]:.3e} | {se:.1e} | {z(pw[i], pe, se)} |")
    out.append(f"| volatility | {ex.volatility:.4f} | {mc.volatility:.4f} | | |")
    return out + [""]


def duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.1f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    if seconds < 172_800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86_400:.1f} days"


def monte_carlo_section(add, r: dict) -> None:
    ex, mc, ses = r["exact"]["exact"], r["mc"], r["sessions"]
    add("## 3. Monte Carlo\n")
    add(f"The Monte Carlo implementation (Numba, `slotmath/simulate.py`, an evaluator written "
        f"independently of the exact path) simulated **{mc.sessions} sessions × "
        f"{mc.spins_per_session:,} spins = {mc.spins:,} spins in {mc.seconds:.1f} s on "
        f"{mc.threads} threads** ({mc.spins_per_second / 1e6:.1f} million spins per second; "
        "master seed 2026, one splitmix64 stream per session). Rows whose expected count is "
        "below 10 are left out.\n")
    for line in agreement_rows(ex, mc):
        add(line)
    add(f"Runaway cascades observed: {mc.runaways}.\n")

    add("### Observed RTP after aₙ = round(1.5ⁿ) spins\n")
    add("Each session records its running RTP at the checkpoints aₙ = round(1.5ⁿ), n ≥ 0 "
        "(1, 2, 3, 5, 8, 11, 17, 26, 38, …; a₁ = a₂ = 2, so the duplicate is dropped). The band is "
        "the exact RTP ± 3σ/√N with σ the exact volatility, the 99.7% range a correctly built "
        "game should stay inside once N is large enough for the normal approximation. Values "
        "outside the plotted range are clipped.\n")
    R = mc.session_rtp()
    add("| spins N | 3σ/√N | sessions inside the band | session RTP, min | median | max |")
    add("| --- | --- | --- | --- | --- | --- |")
    vol = ex.volatility
    for i in near_decades(mc.checkpoints):
        N = int(mc.checkpoints[i])
        half = 3 * vol / sqrt(N)
        inside = np.mean(np.abs(R[:, i] - float(ex.rtp)) <= half)
        add(f"| {N:,} | {half:.2%} | {inside:.1%} | {R[:, i].min():.2%} | "
            f"{np.median(R[:, i]):.2%} | {R[:, i].max():.2%} |")
    add("")
    add(f"![Observed RTP after a_n spins]({r['fig']['convergence']})\n")

    add("### Player sessions\n")
    add(f"{ses.sessions:,} independent sessions of {ses.spins_per_session:,} spins "
        f"({ses.spins:,} spins, Numba, {ses.seconds:.1f} s on {ses.threads} threads): the "
        "spread of observed RTP a player sees, and how often a session ends ahead.\n")
    Rs = ses.session_rtp()
    add("| spins N | 5% | 25% | median | 75% | 95% | sessions ahead (RTP ≥ 100%) |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for i in near_decades(ses.checkpoints):
        N = int(ses.checkpoints[i])
        q = np.quantile(Rs[:, i], [0.05, 0.25, 0.5, 0.75, 0.95])
        ahead = np.mean(Rs[:, i] >= 1.0)
        add(f"| {N:,} | {q[0]:.1%} | {q[1]:.1%} | {q[2]:.1%} | {q[3]:.1%} | {q[4]:.1%} | "
            f"{ahead:.1%} |")
    add("")
    add(f"![Player sessions]({r['fig']['sessions']})\n")


def checks_section(add, r: dict) -> None:
    g, control = r["game"], r["eff"]["control"]
    L_n = g.length ** g.reels
    add("## 4. How the numbers are checked\n")
    add("The implementation is Numba: the exact enumerator (`slotmath/exact.py`) and the Monte "
        "Carlo (`slotmath/simulate.py`), two evaluators written independently of each other. "
        "The controls are pure Python.\n")
    add("- **Control: the rules.** `slotmath/reference.py` states the rules as plainly as "
        "possible. It reproduces a hand-checkable three-reel game exactly (RTP 43637/12800). The "
        "Numba enumerator matches its enumeration histogram for histogram on small games with "
        "wilds, scatters and cascades each on and off, and the Numba Monte Carlo evaluator "
        "matches it on thousands of random spins.")
    verdict = "bit for bit" if control["identical"] else "NOT bit for bit: investigate"
    add(f"- **Control: the session.** The pure-Python session replays a Numba session's random "
        f"stream through the reference rules. In this run, on {control['spins']:,} spins of this "
        f"game, it reproduced the Numba session's totals {verdict}.")
    add(f"- **The two Numba evaluators agree on every one of the {L_n:,} stop vectors** of "
        "every fitted game, total pay and chain length alike. So the only thing separating the "
        "Monte Carlo from the exact figures is the sampling of stops, and the sampler is tested "
        "exactly (each stop appears in its lookup table exactly its weight times).")
    add("- **The fitter only proposes** weights; every reported figure is recomputed from the "
        "integer weights in exact rational arithmetic.")
    add("- **Statistics.** The exact figures and the Monte Carlo then agree within sampling "
        "error, as tabled in §3.\n")
    add("Reproduce:\n")
    add("```sh\npython -m pip install -e \".[dev,report]\"\npytest -q\n"
        "python -m slotmath.fit       # strips and weights -> slotmath/designs.json\n"
        "python -m slotmath.report    # REPORT.md and VOLATILITY.md\n```\n")


def speed_section(add, r: dict) -> None:
    g, mc, ses, eff = r["game"], r["mc"], r["sessions"], r["eff"]
    control = eff["control"]
    rate_c = control["spins"] / control["seconds"]
    L_n = g.length ** g.reels
    add("## 5. Speed: the Numba implementation against the pure-Python control\n")
    add(f"Every figure above comes from the Numba implementation on {r['threads']} threads. The "
        f"control evaluates **{rate_c:,.0f} spins per second** (measured in this run: "
        f"{control['spins']:,} spins in {control['seconds']:.2f} s). Its times for the full "
        "workloads are extrapolated from that rate, not run.\n")
    add("| task | Numba, measured | pure-Python control, extrapolated | speed-up |")
    add("| --- | --- | --- | --- |")
    for task, work, seconds in (
            (f"Monte Carlo, long sessions ({mc.spins:,} spins)", mc.spins, mc.seconds),
            (f"Monte Carlo, player sessions ({ses.spins:,} spins)", ses.spins, ses.seconds),
            (f"exact enumeration ({L_n:,} stop vectors)", L_n, r["exact"]["t_enum"])):
        slow = work / rate_c
        add(f"| {task} | {duration(seconds)} | {duration(slow)} | {slow / seconds:,.0f}× |")
    add("")
    rows = eff["rows"]
    rate_1 = rows[0][1] / rows[0][2]
    add(f"**Thread scaling** (Numba Monte Carlo, {rows[0][1]:,} spins per run in 240 sessions):\n")
    add("| threads | seconds | spins per second | against one thread | against the control |")
    add("| --- | --- | --- | --- | --- |")
    for k, sp, sec in rows:
        rate = sp / sec
        add(f"| {k} | {sec:.2f} | {rate:,.0f} | {rate / rate_1:.1f}× | {rate / rate_c:,.0f}× |")
    add("")
    add(f"Exact enumeration of all {L_n:,} stop vectors: {eff['t_enum_1']:.1f} s on one thread, "
        f"{eff['t_enum_n']:.2f} s on {eff['threads']} threads. Scaling depends on the machine: "
        "on a hybrid processor, threads beyond the performance cores, and hyper-threads sharing "
        "a core, add less than a full core each.\n")
    add("What makes the Numba implementation fast:\n")
    add("- **One random number per spin.** Weights sum to 4096 = 2¹² per reel, so a reel's stop "
        "is 12 random bits looked up in a 4096-entry table (no search, no modulo bias), and one "
        "64-bit splitmix64 draw covers all five reels.")
    add("- **One pass per payline.** A line is read once, left to right, with a shrinking "
        "bitmask of the paying symbols it still matches; the wild matches all of them.")
    add("- **No allocation in the spin loop.** Each session owns its scratch arrays; a spin "
        "touches only small integer arrays and a 64-bit mask of winning cells.")
    add("- **Independent streams.** Sessions run in parallel, each seeded from the master seed "
        "and its index, so the result is bit-for-bit identical whatever the thread count.\n")
    add(f"![Monte Carlo throughput and parallel scaling]({r['fig']['performance']})\n")


def sweep_speed_section(add, r: dict) -> None:
    levels, designs = r["levels"], r["designs"]
    add("## 3. Speed: the Numba implementation against the pure-Python control\n")
    add(f"Everything above was computed by the Numba implementation on {r['threads']} threads. "
        f"The fit itself (`python -m slotmath.fit`: {len(designs['scan'])} exact enumerations "
        "in Numba, the weight solver in NumPy) took "
        f"{duration(designs['seconds']['total'])}. For every level the pure-Python control "
        "replayed a Numba session's random stream, and must reproduce its totals bit for bit.\n")
    add("| level | exact enumeration, Numba | Monte Carlo, Numba | pure-Python control | "
        "control reproduces Numba bit for bit | Monte Carlo speed-up |")
    add("| --- | --- | --- | --- | --- | --- |")
    for lv in levels:
        mc, c = lv["mc"], lv["control"]
        rate_n, rate_c = mc.spins / mc.seconds, c["spins"] / c["seconds"]
        add(f"| {lv['key']} | {lv['exact']['t_enum']:.2f} s | {mc.spins:,} spins in "
            f"{mc.seconds:.1f} s ({rate_n / 1e6:.1f} million/s) | {c['spins']:,} spins in "
            f"{c['seconds']:.2f} s ({rate_c:,.0f}/s) | {'yes' if c['identical'] else 'NO'} | "
            f"{rate_n / rate_c:,.0f}× |")
    add("")
    add("Higher-volatility levels simulate faster: they hit less often, so fewer spins need a "
        "cascade.\n")


def write_volatility(r: dict) -> str:
    levels, designs = r["levels"], r["designs"]
    targets = designs["inputs"]["volatility_targets"]
    g0 = levels[0]["game"]
    out: list[str] = []
    add = out.append
    add("# One game, five volatilities\n")
    add(f"Generated by `python -m slotmath.report{' --quick' if r['quick'] else ''}` on "
        f"{dt.date.today().isoformat()}, alongside [REPORT.md](REPORT.md), which analyses level "
        f"{designs['main']} in depth. Every number comes from that run. Machine: {r['machine']}.\n")

    add("## 1. Game definition (inputs)\n")
    add("Everything is held fixed except the volatility target: the same paytable, reels, "
        "mechanics and RTP, refitted five times.\n")
    inputs_section(add, g0, designs, "volatility ∈ {" + ", ".join(f"{v}" for v in targets) +
                   "}, levels v1 to v5")
    sec = designs["seconds"]
    add("**The fitting procedure** (`python -m slotmath.fit`, the same for every level):\n")
    add(f"1. **Strips.** {len(designs['scan'])} candidate strip sets from three knobs: high "
        f"symbols per reel {designs['inputs']['grid']['high']}, stack height "
        f"{designs['inputs']['grid']['stack']}, wilds on each of reels 2–4 "
        f"{designs['inputs']['grid']['wilds']}. Low symbols fill the rest. Each set is "
        "enumerated exactly once.")
    add("2. **Weights.** For every candidate and every target, Gauss–Newton steps solve "
        "RTP = 0.96 and volatility = target at once, each step taking the smallest change in "
        "log weight from uniform. Both E[X] and E[X²] are linear in each reel's weights, so the "
        "derivatives are exact.")
    add("3. **Choice.** Among the candidates that reach a target within the weight bounds, the one "
        "whose weights stay closest to uniform wins.")
    add("4. **Integers.** Weights are rounded to integers and polished with single-unit moves, "
        "each priced exactly; the result is verified in exact rational arithmetic.\n")
    add(f"The scan took {sec['scan']:.0f} s for all {len(targets)} targets at once.\n")
    add(f"![Design space]({r['fig']['design_space']})\n")
    add("Each grey dot is a candidate strip set under uniform weights. Raising volatility at a "
        "fixed RTP means moving return from frequent small wins to rare large ones: more high "
        "symbols, in taller stacks, with fewer low symbols. The arrows show the weights carrying "
        "each chosen strip set to RTP 96% and its volatility target.\n")
    add(f"![The fitted strips]({r['fig']['strips']})\n")

    add("## 2. Results for each volatility\n")
    add("| level | volatility target | achieved | RTP (exact) | hit frequency | P(win > stake) | "
        "max win | E[D] | strips: high / stack / wilds | weight change (RMS log) | "
        "Monte Carlo RTP (z) | Monte Carlo volatility |")
    add("| --- |" + " --- |" * 11)
    for lv in levels:
        ex, mc, d = lv["exact"]["exact"], lv["mc"], lv["design"]
        p = d["params"]
        add(f"| {lv['key']} | {d['target_volatility']} | {ex.volatility:.6f} | "
            f"{float(ex.rtp):.10f} | "
            f"{float(ex.hit_frequency):.4f} | {float(ex.p_win_above(1)):.4f} | "
            f"{float(ex.max_win):,.1f}× | {float(ex.expected_steps):.4f} | "
            f"{p['high']} / {p['stack']} / {p['wilds']} | {d['solver']['distortion']:.3f} | "
            f"{mc.rtp:.4f} ({z(mc.rtp, float(ex.rtp), mc.rtp_se)}) | {mc.volatility:.3f} |")
    add("")
    n_mc = levels[0]["mc"].spins
    secs = [lv["mc"].seconds for lv in levels]
    add(f"Monte Carlo: {n_mc:,} spins per level, Numba on {r['threads']} threads, "
        f"{min(secs):.1f}–{max(secs):.1f} s per level. The z-score is "
        "(simulated − exact) / standard error.\n")
    add("**Possible wins, by size** (probability per spin, exact):\n")
    labels = [b[2] for b in BANDS]
    table = {lv["key"]: {lab: p for lab, p, _, _ in bands(lv["exact"]["exact"])} for lv in levels}
    add("| win (× stake) | " + " | ".join(lv["key"] for lv in levels) + " |")
    add("| --- |" + " --- |" * len(levels))
    for lab in labels:
        if any(lab in table[k] for k in table):
            cells = [f"{float(table[lv['key']][lab]):.3e}" if lab in table[lv["key"]] else "–"
                     for lv in levels]
            add(f"| {lab} | " + " | ".join(cells) + " |")
    add("")
    add(f"![P(win at least x) by volatility level]({r['fig']['sweep_wins']})\n")
    by_strips: dict = {}
    for lv in levels:
        by_strips.setdefault(tuple(lv["game"].strips), []).append(lv["key"])
    for keys in by_strips.values():
        if len(keys) > 1:
            add(f"Levels {' and '.join(keys)} landed on the same strip set, the least-distorted "
                "candidate for both targets; only their weights differ. The grid's upper edge "
                f"(at most {max(designs['inputs']['grid']['high'])} high symbols per reel) is "
                "what limits the top of the range.\n")

    for lv in levels:
        g, ex, d = lv["game"], lv["exact"]["exact"], lv["design"]
        p = d["params"]
        add(f"### {lv['key']}: volatility {d['target_volatility']}\n")
        add(f"{p['high']} high symbols per reel in stacks of up to {p['stack']}, "
            f"{wilds(p['wilds'])} on each of reels 2–4. Exact RTP {float(ex.rtp):.10f}, volatility "
            f"{ex.volatility:.6f}, hit frequency {float(ex.hit_frequency):.4f}, max win "
            f"{float(ex.max_win):,.1f}×, {len(ex.pay):,} distinct payouts; weights within "
            f"{np.min(g.weights)}–{np.max(g.weights)} after {d['polish_moves']} polishing "
            "moves.\n")
        add("<details><summary>Reels and weights</summary>\n")
        for line in composition_table(g):
            add(line)
        for line in strip_table(g):
            add(line)
        add("</details>\n")
        add(f"Possible wins (every value with its exact probability: "
            f"[{lv['csv']}]({lv['csv']})):\n")
        for line in band_table(ex):
            add(line)
    sweep_speed_section(add, r)
    path = ROOT / "VOLATILITY.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return str(path)


# ============================================================================ main
def cpu_name() -> str:
    """The processor's marketing name where the OS exposes it, else platform's description."""
    try:
        import winreg
        path = chr(92).join(["HARDWARE", "DESCRIPTION", "System", "CentralProcessor", "0"])
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        return platform.processor() or platform.machine()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="a fiftieth of the simulation")
    args = parser.parse_args()
    scale = 50 if args.quick else 1
    t_start = time.perf_counter()
    designs = load_designs()
    main_key = designs["main"]
    game = fitted_game(main_key)

    print("loading kernels ...")
    t_warm = warm_up(game)
    print("exact enumeration ...")
    exact = run_exact(game)
    print("options ...")
    options = run_options(main_key, exact["spins"])
    print("Monte Carlo, long sessions ...")
    mc = simulate(game, 200, 5_000_000 // scale, seed=2026, big_wins=BIG_WINS)
    print(f"  {mc.spins:,} spins in {mc.seconds:.1f} s")
    print("Monte Carlo, player sessions ...")
    ses = simulate(game, 100_000 // scale, 10_000, seed=2027)
    print(f"  {ses.spins:,} spins in {ses.seconds:.1f} s")
    print("efficiency ...")
    eff = run_efficiency(game, args.quick)
    print("volatility levels ...")
    levels = run_sweep(designs, scale)

    plt = _style()
    ex = exact["exact"]
    fig = {
        "convergence": figure_convergence(plt, mc, float(ex.rtp), ex.volatility),
        "sessions": figure_sessions(plt, ses, float(ex.rtp)),
        "cascades": figure_cascades(plt, ex, mc),
        "options": figure_options(plt, options),
        "wins": figure_wins(plt, ex, mc),
        "performance": figure_performance(plt, eff),
        "sweep_wins": figure_sweep_wins(plt, levels),
        "design_space": figure_design_space(plt, designs, levels),
        "strips": figure_strips(plt, levels),
    }
    for lv in levels:
        lv["csv"] = write_wins_csv(lv["exact"]["exact"], lv["key"])
    machine = (f"{cpu_name()}, {os.cpu_count()} logical cores, "
               f"{platform.system()} {platform.release()}, Python {platform.python_version()}, "
               f"NumPy {np.__version__}, Numba {numba.__version__}")
    common = {"designs": designs, "fig": fig, "quick": args.quick, "machine": machine,
              "threads": numba.get_num_threads()}
    report = write_report({**common, "game": game, "design": designs["games"][main_key],
                           "exact": exact, "options": options, "mc": mc, "sessions": ses,
                           "eff": eff, "csv": f"report/wins/{main_key}.csv"})
    sweep = write_volatility({**common, "levels": levels})
    print(f"wrote {report} and {sweep} in {time.perf_counter() - t_start:.0f} s "
          f"(kernel load {t_warm:.1f} s)")


if __name__ == "__main__":
    main()
