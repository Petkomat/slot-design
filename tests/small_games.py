"""Games small enough for the pure-Python reference to enumerate: 6 ** 5 = 7,776 stop vectors."""

from slotmath.game import LINES_20, THREE_LINE_PAY, THREE_VERTICAL_PAY, Game, eight_symbol_game


def small_three_symbol_game(**options) -> Game:
    """Five reels of six positions: 7,776 stop vectors, few enough for the Python reference."""
    strips = ("AB-CWC", "CCBW-A", "BCC-WA", "C-ABCW", "ACB-CC")
    weights = ((700, 680, 690, 676, 650, 700), (600, 800, 700, 696, 600, 700),
               (683, 683, 682, 683, 682, 683), (1000, 500, 700, 696, 600, 600),
               (650, 700, 690, 676, 680, 700))
    g = Game(strips=strips, weights=weights, lines=LINES_20, line_pay=THREE_LINE_PAY,
             vertical_pay=THREE_VERTICAL_PAY)
    return g.with_options(**options) if options else g


def small_eight_symbol_game(**options) -> Game:
    """The eight-symbol paytable on five reels of six positions, wilds and scatters dense enough
    that every rule fires often."""
    strips = (("H1", "L1", "S", "L5", "H3", "L2"), ("W", "H1", "L1", "S", "L4", "H2"),
              ("L1", "W", "H1", "S", "H2", "L3"), ("H1", "L1", "W", "S", "L5", "L1"),
              ("S", "H1", "L1", "L2", "H3", "W"))
    weights = ((700, 680, 690, 676, 650, 700), (600, 800, 700, 696, 600, 700),
               (683, 683, 682, 683, 682, 683), (1000, 500, 700, 696, 600, 600),
               (650, 700, 690, 676, 680, 700))
    return eight_symbol_game(strips, weights, **options)
