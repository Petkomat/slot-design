"""Totals of a seeded spin stream, for the browser demo's evaluator to be checked against.

The demo under ``web/`` re-implements the rules in TypeScript, which makes it a fourth evaluator
of a game that already has three. It is held to the same standard as the others: this module
records what a known random stream pays under ``reference.py``, and the TypeScript regenerates
the same stops from the same seed and must reproduce every total exactly.

Exactly, not within a tolerance. The obvious alternative, comparing a million-spin empirical RTP
against the fitted 0.96, has a standard error of volatility / sqrt(spins) = 0.0045 at a million
spins: it would miss any error worth less than about 1% of RTP, and fail by chance about one run
in twenty. Agreement on totals catches a single mispaid payline and never fails by chance.

The stream is defined in ``simulate.py``: ``stream_state(seed, stream)`` seeds splitmix64, and
each 64-bit draw supplies five 12-bit indices into per-reel lookup tables in which stop ``t``
appears ``weight[t]`` times. All of that is arithmetic TypeScript can repeat.

Run ``python -m slotmath.webfixture`` after refitting; the digest below makes a stale fixture a
test failure rather than a silent disagreement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .game import WEIGHT_TOTAL, Game, fitted_game, load_designs
from .simulate import (
    lookup_tables,
    simulate,
    simulate_python,
    splitmix64_python,
    stream_state_python,
)

WEB_SRC = Path(__file__).resolve().parents[1] / "web" / "src"
FIXTURE_FILE = WEB_SRC / "fixture.json"
GAME_FILE = WEB_SRC / "game.json"
FAST_SPINS = 20_000
SLOW_SPINS = 1_000_000
SEED = 2026
STREAM = 0


def game_parameters(game: Game) -> dict:
    """The rule parameters the demo needs, in the form ``reference.py`` itself reads them.

    Symbols become their index, and every pay is in units, so the TypeScript evaluator is handed
    the same numbers the Python one works from and implements only the algorithm itself. A
    paytable transcribed by hand into another language is a silent error waiting to happen.
    """
    a = game.arrays()
    return {
        "symbols": list(game.symbols),
        "wild": int(a["flags"][1]),
        "scatter": int(a["flags"][5]),
        "paying": [int(s) for s in a["paying"]],
        "reels": game.reels,
        "rows": game.rows,
        "length": game.length,
        "lines": [list(line) for line in game.lines],
        "linePay": [[int(v) for v in row] for row in a["line_pay"]],
        "scatterPay": [int(v) for v in a["scatter_pay"]],
        "ladder": [int(v) for v in game.ladder],
        "maxSteps": game.max_steps,
        "stakeUnits": game.stake_units,
        "weightTotal": WEIGHT_TOTAL,
    }


def design_digest(design: dict) -> str:
    """A short hash of the strips and weights, so a refit invalidates the recorded totals."""
    payload = json.dumps([design["strips"], design["weights"]], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stream_sample(game: Game, draws: int = 8) -> dict:
    """The first few raw 64-bit outputs and the stop vectors they produce.

    Pinned separately from the totals so that a disagreement says which half is wrong: the
    generator, or the evaluator reading it. The outputs exceed 2**53, so they are recorded as
    decimal strings.
    """
    lut = lookup_tables(game.weights)
    state = stream_state_python(SEED, STREAM)
    outputs, stops = [], []
    for _ in range(draws):
        state, z = splitmix64_python(state)
        outputs.append(str(z))
        stops.append([int(lut[r, (z >> (12 * r)) & 0xFFF]) for r in range(game.reels)])
    return {"seed": SEED, "stream": STREAM, "draws": outputs, "stops": stops}


def _steps_histogram(counts: dict[int, int]) -> list[int]:
    """Cascade-step counts as a dense list, index = number of winning windows in the spin."""
    return [counts.get(i, 0) for i in range(max(counts) + 1)]


def run_python(game: Game, spins: int) -> dict:
    """``spins`` spins of the reference evaluator on the standard stream."""
    out = simulate_python(game, spins, seed=SEED, stream=STREAM)
    return {"seed": SEED, "stream": STREAM, "spins": spins,
            "pay_sum": int(out["pay_sum"]), "pay_sq": int(out["pay_sq"]),
            "hits": int(out["hits"]), "steps": _steps_histogram(out["steps"])}


def run_compiled(game: Game, spins: int) -> dict:
    """The same stream through the compiled evaluator: one session is stream 0."""
    r = simulate(game, 1, spins, seed=SEED)
    hist = r.step_hist.sum(axis=0).tolist()
    while len(hist) > 1 and hist[-1] == 0:
        hist.pop()
    return {"seed": SEED, "stream": STREAM, "spins": spins,
            "pay_sum": int(r.pay_sum.sum()), "pay_sq": int(r.pay_sq.sum()),
            "hits": int(r.hits.sum()), "steps": [int(c) for c in hist]}


def build(name: str = "main") -> tuple[dict, dict]:
    """The fixture and the rule parameters: a short run from pure Python and a long one from
    the compiled twin.

    The short run is produced both ways and the two are required to match before the long run,
    produced only by the compiled evaluator, is trusted.
    """
    designs = load_designs()
    key = designs["main"] if name == "main" else name
    design = designs["games"][key]
    game = fitted_game(key)

    fast_py = run_python(game, FAST_SPINS)
    fast_compiled = run_compiled(game, FAST_SPINS)
    if fast_py != fast_compiled:
        raise AssertionError(f"evaluators disagree on the stream:\n{fast_py}\n{fast_compiled}")
    slow = run_compiled(game, SLOW_SPINS)

    return {
        "design": key,
        "digest": design_digest(design),
        "stake_units": game.stake_units,
        "rtp_exact": design["rtp"],
        "volatility_exact": design["volatility"],
        "sample": stream_sample(game),
        "runs": {"fast": fast_py, "slow": slow},
    }, game_parameters(game)


def main() -> None:
    fixture, parameters = build()
    WEB_SRC.mkdir(parents=True, exist_ok=True)
    FIXTURE_FILE.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    GAME_FILE.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    for label, run in fixture["runs"].items():
        rtp = run["pay_sum"] / run["spins"] / fixture["stake_units"]
        print(f"{label:>5}: {run['spins']:>9,} spins, pay_sum {run['pay_sum']:>14,}, "
              f"session RTP {rtp:.6f}")
    print(f"written to {FIXTURE_FILE} and {GAME_FILE}")


if __name__ == "__main__":
    main()
