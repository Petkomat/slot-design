"""Game definition: strips, stop weights, paylines, paytable and mechanic options.

A spin draws one stop per reel, independently, reel i landing on position t with probability
weights[i][t] / 4096. The window shows `rows` consecutive positions from the stop, top row first.

Symbols are named; a strip is a sequence of names (a plain string is read one character per
position). A symbol is *paying* if it has a line pay, the *wild* substitutes for every paying
symbol on a line, and the *scatter* pays by how many are visible anywhere. Any other symbol is a
blank.

Units. Every payout is held as an integer number of *units*, a unit being one fifth of a line
bet, so fractional pays such as a vertical of a fifth of a line bet stay integral. The stake is
one line bet per payline, so a spin costs ``5 * len(lines)`` units. Line pays are quoted in line
bets and scatter pays in multiples of the stake, as paytables usually quote them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

UNITS_PER_LINE_BET = 5
WEIGHT_TOTAL = 4096  # 2**12: one 64-bit draw yields five unbiased 12-bit stops

# A conventional 20-line set for a 5x3 window: row offset per reel, top row = 0.
LINES_20: tuple[tuple[int, ...], ...] = (
    (1, 1, 1, 1, 1), (0, 0, 0, 0, 0), (2, 2, 2, 2, 2), (0, 1, 2, 1, 0), (2, 1, 0, 1, 2),
    (0, 0, 1, 2, 2), (2, 2, 1, 0, 0), (1, 0, 0, 0, 1), (1, 2, 2, 2, 1), (0, 1, 1, 1, 0),
    (2, 1, 1, 1, 2), (1, 0, 1, 2, 1), (1, 2, 1, 0, 1), (0, 1, 0, 1, 0), (2, 1, 2, 1, 2),
    (1, 1, 0, 1, 1), (1, 1, 2, 1, 1), (0, 0, 2, 0, 0), (2, 2, 0, 2, 2), (0, 2, 2, 2, 0),
)


@dataclass(frozen=True)
class Game:
    """A slot game: what is on the reels, how likely each stop is, and what pays."""

    strips: tuple[tuple[str, ...], ...]
    weights: tuple[tuple[int, ...], ...]
    lines: tuple[tuple[int, ...], ...]
    line_pay: dict[str, dict[int, int | Fraction]]          # in line bets, by run length
    symbols: tuple[str, ...] = ("-", "A", "B", "C", "W")
    wild: str | None = "W"
    scatter: str | None = None
    scatter_pay: dict[int, int | Fraction] = field(default_factory=dict)  # stakes, by count
    vertical_pay: dict[str, int | Fraction] = field(default_factory=dict)  # in line bets
    rows: int = 3
    wilds: bool = True
    scatters: bool = True
    verticals: bool = False
    cascades: bool = True
    ladder: tuple[int, ...] = (1, 2, 3, 5)  # multiplier for cascade step 0, 1, 2, 3+
    max_steps: int = 100  # a chain this long is treated as a runaway

    def __post_init__(self) -> None:
        object.__setattr__(self, "strips", tuple(tuple(s) for s in self.strips))
        object.__setattr__(self, "weights", tuple(tuple(int(v) for v in w) for w in self.weights))
        self.validate()

    # ------------------------------------------------------------------ shape
    @property
    def reels(self) -> int:
        return len(self.strips)

    @property
    def length(self) -> int:
        return len(self.strips[0])

    @property
    def stake_units(self) -> int:
        return UNITS_PER_LINE_BET * len(self.lines)

    @property
    def paying(self) -> tuple[str, ...]:
        """Paying symbols, in alphabet order."""
        return tuple(s for s in self.symbols if s in self.line_pay)

    def with_options(self, **changes) -> "Game":
        """The same game with some options changed, e.g. ``g.with_options(wilds=False)``."""
        return replace(self, **changes)

    def scatter_units(self, count: int) -> int:
        """Scatter pay in units for ``count`` visible scatters; the largest count listed also
        pays for any higher count."""
        if not (self.scatters and self.scatter and self.scatter_pay):
            return 0
        eligible = [c for c in self.scatter_pay if c <= count]
        if not eligible:
            return 0
        return int(Fraction(self.scatter_pay[max(eligible)]) * self.stake_units)

    # ------------------------------------------------------------- validation
    def validate(self) -> None:
        n, L = self.reels, self.length
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbol names must be unique")
        if any(len(s) != L for s in self.strips):
            raise ValueError("every reel must have the same length L")
        if any(ch not in self.symbols for s in self.strips for ch in s):
            raise ValueError(f"strips may only use the symbols {self.symbols!r}")
        for role, name in (("wild", self.wild), ("scatter", self.scatter)):
            if name is not None and name not in self.symbols:
                raise ValueError(f"the {role} {name!r} is not one of the symbols")
        for s in (*self.line_pay, *self.vertical_pay):
            if s not in self.symbols or s in (self.wild, self.scatter):
                raise ValueError(f"{s!r} cannot carry a line or vertical pay")
        if len(self.weights) != n or any(len(w) != L for w in self.weights):
            raise ValueError("one weight per strip position is required")
        for i, w in enumerate(self.weights):
            if min(w) < 1:
                raise ValueError(f"reel {i + 1}: every stop needs a positive weight")
            if sum(w) != WEIGHT_TOTAL:
                raise ValueError(f"reel {i + 1}: weights sum to {sum(w)}, not {WEIGHT_TOTAL}")
        if L < self.rows:
            raise ValueError("a strip must be at least as long as the window")
        for line in self.lines:
            if len(line) != n or any(not 0 <= r < self.rows for r in line):
                raise ValueError(f"bad payline {line}")
        for table in (*self.line_pay.values(), self.vertical_pay):
            for v in table.values():
                if (Fraction(v) * UNITS_PER_LINE_BET).denominator != 1:
                    raise ValueError(f"pay {v} is not a whole number of fifths of a line bet")
        for count, v in self.scatter_pay.items():
            if count < 1 or (Fraction(v) * self.stake_units).denominator != 1:
                raise ValueError(f"bad scatter pay {count}: {v}")
        if not self.ladder or min(self.ladder) < 1:
            raise ValueError("the multiplier ladder needs at least one positive entry")

    # ----------------------------------------------------- arrays for kernels
    def arrays(self) -> dict[str, np.ndarray]:
        """The game as plain integer arrays, the form the compiled kernels take. Symbols become
        their index in ``symbols``; a switched-off wild or scatter becomes index -1."""
        n, m = self.reels, len(self.symbols)
        index = {s: i for i, s in enumerate(self.symbols)}
        strips = np.array([[index[ch] for ch in s] for s in self.strips], dtype=np.int64)
        line_pay = np.zeros((m, n + 1), dtype=np.int64)
        for s, table in self.line_pay.items():
            for k, pay in table.items():
                line_pay[index[s], k] = int(Fraction(pay) * UNITS_PER_LINE_BET)
        vertical_pay = np.zeros(m, dtype=np.int64)
        for s, pay in self.vertical_pay.items():
            vertical_pay[index[s]] = int(Fraction(pay) * UNITS_PER_LINE_BET)
        scatter_pay = np.array([self.scatter_units(c) for c in range(n * self.rows + 1)],
                               dtype=np.int64)
        wild = index[self.wild] if self.wilds and self.wild is not None else -1
        scatter = index[self.scatter] if self.scatters and self.scatter is not None else -1
        return {
            "strips": strips,
            "weights": np.array(self.weights, dtype=np.int64),
            "lines": np.array(self.lines, dtype=np.int64),
            "line_pay": line_pay,
            "vertical_pay": vertical_pay,
            "scatter_pay": scatter_pay,
            "paying": np.array([index[s] for s in self.paying], dtype=np.int64),
            "ladder": np.array(self.ladder, dtype=np.int64),
            "flags": np.array([self.rows, wild, int(self.verticals), int(self.cascades),
                               self.max_steps, scatter], dtype=np.int64),
        }


# --------------------------------------------------------------------- weights
def uniform_weights(length: int, reels: int) -> tuple[tuple[int, ...], ...]:
    """Equal weight on every stop; needs ``length`` to divide 4096."""
    if WEIGHT_TOTAL % length:
        raise ValueError(f"uniform weights need L to divide {WEIGHT_TOTAL}")
    return tuple((WEIGHT_TOTAL // length,) * length for _ in range(reels))


# ------------------------------------------------- the eight-symbol game (fitted)
EIGHT_SYMBOLS = ("H1", "H2", "H3", "L1", "L2", "L3", "L4", "L5", "W", "S")
HIGH = ("H1", "H2", "H3")
LOW = ("L1", "L2", "L3", "L4", "L5")

# Line pays in line bets, for 3, 4 and 5 of a kind from reel 1. W substitutes for every paying
# symbol and has no pay of its own (a line of five wilds pays as H1). S pays in multiples of the
# stake by how many are visible anywhere, once, on the window the cascade chain ends on.
EIGHT_LINE_PAY: dict[str, dict[int, int]] = {
    "H1": {3: 25, 4: 100, 5: 500},
    "H2": {3: 15, 4: 60, 5: 250},
    "H3": {3: 10, 4: 40, 5: 150},
    "L1": {3: 5, 4: 20, 5: 100},
    "L2": {3: 5, 4: 15, 5: 75},
    "L3": {3: 4, 4: 12, 5: 60},
    "L4": {3: 3, 4: 10, 5: 50},
    "L5": {3: 2, 4: 8, 5: 40},
}
EIGHT_SCATTER_PAY: dict[int, int] = {3: 2, 4: 10, 5: 50}   # 5 or more pays 50x

DESIGNS_FILE = Path(__file__).with_name("designs.json")


def eight_symbol_game(strips, weights, **options) -> Game:
    """The eight-symbol 5x3, 20-line game on the given strips and weights."""
    game = Game(strips=strips, weights=weights, lines=LINES_20, line_pay=EIGHT_LINE_PAY,
                symbols=EIGHT_SYMBOLS, wild="W", scatter="S", scatter_pay=EIGHT_SCATTER_PAY)
    return game.with_options(**options) if options else game


def load_designs() -> dict:
    """The fitted designs written by ``python -m slotmath.fit``."""
    return json.loads(DESIGNS_FILE.read_text(encoding="utf-8"))


def fitted_game(name: str = "main", **options) -> Game:
    """A fitted eight-symbol game: ``"main"``, or a volatility level ``"v1"`` .. ``"v5"``."""
    designs = load_designs()
    key = designs["main"] if name == "main" else name
    d = designs["games"][key]
    return eight_symbol_game(d["strips"], d["weights"], **options)


# ------------------------------------------- the three-symbol game and the toy
# Kept as fixtures: the tests pin their exact figures, which guards every evaluator. The
# three-symbol game's strips carry blanks, as classic three-reel machines do; five-reel video
# slots do not, which is why the fitted game above has none.
THREE_LINE_PAY = {"A": {3: 25, 4: 125, 5: 500}, "B": {3: 5, 4: 25, 5: 100},
                  "C": {3: 1, 4: 5, 5: 20}}
THREE_VERTICAL_PAY = {"A": Fraction(5), "B": Fraction(1), "C": Fraction(1, 5)}

_Y = "AA-BBB-CCCC-W-CCC-BBBB-CCCC-AA-W"   # reels 2 and 4 (reel 4 reversed)
_X = _Y.replace("W", "-")                 # reels 1 and 5 (reel 5 reversed): no wild
_Z = "A-CCCC-BB-W-CCC-AA-BBB-CC-WCCBAB"   # reel 3
THREE_STRIPS: tuple[str, ...] = (_X, _Y, _Z, _Y[::-1], _X[::-1])

# Fixed test data: stop weights once tuned to an exact RTP of
# 22136092935839664877 / 23058430092136939520 = 0.9600000021, which the tests pin.
THREE_WEIGHTS = (
    (125, 107, 102, 87, 139, 114, 112, 100, 112, 142, 160, 172, 164, 154, 144, 151,
     135, 135, 119, 121, 139, 116, 112, 101, 112, 142, 120, 115, 122, 147, 149, 126),
    (138, 130, 119, 115, 151, 119, 108, 110, 127, 155, 119, 116, 136, 135, 128, 153,
     152, 143, 133, 136, 150, 121, 110, 110, 126, 155, 147, 130, 142, 104, 83, 95),
    (138, 146, 135, 149, 167, 143, 152, 158, 86, 115, 103, 132, 121, 164, 154, 142,
     149, 128, 129, 113, 159, 111, 111, 143, 115, 108, 105, 94, 84, 110, 100, 132),
    (116, 135, 135, 154, 154, 142, 141, 157, 155, 126, 100, 94, 115, 121, 120, 113,
     153, 126, 112, 129, 136, 127, 133, 156, 155, 126, 100, 116, 140, 136, 89, 84),
    (158, 138, 130, 136, 140, 123, 122, 135, 135, 123, 88, 84, 100, 114, 131, 114,
     135, 154, 172, 159, 142, 124, 122, 135, 135, 123, 88, 102, 117, 138, 130, 149),
)


def three_symbol_game(**options) -> Game:
    """The earlier 5x3, 20-line game: three paying symbols, blanks, verticals available."""
    game = Game(strips=THREE_STRIPS, weights=THREE_WEIGHTS, lines=LINES_20,
                line_pay=THREE_LINE_PAY, vertical_pay=THREE_VERTICAL_PAY)
    return game.with_options(**options) if options else game


def toy_game() -> Game:
    """A three-reel, eight-position game: 512 spins, every figure checkable by hand."""
    return Game(
        strips=("ABCCCBW-", "CCCABWB-", "BCCCABW-"),
        weights=uniform_weights(8, 3),
        lines=((0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 1, 2), (2, 1, 0)),
        line_pay={"A": {3: 25}, "B": {3: 5}, "C": {3: 1}},
        vertical_pay=THREE_VERTICAL_PAY,
        verticals=True,
    )
