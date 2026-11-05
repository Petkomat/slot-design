// The rules, re-implemented for the browser. The twin of `slotmath/reference.py`, function for
// function: `evaluateWindow` is `evaluate_window`, `play` is `play`.
//
// This is a fourth evaluator of a game that already has three in Python, which is exactly what
// this project otherwise refuses to allow. It is held to the same standard as the others:
// `test/agreement.test.ts` replays a seeded stream recorded by `slotmath/webfixture.py` and
// requires every total to match exactly, not within a tolerance.
//
// The paytable, paylines and multiplier ladder are not restated here. They are inputs, read
// from `game.json` exactly as `reference.py` reads them from `game.arrays()`; what is written
// independently is the algorithm. Verticals, an option this game leaves off, are omitted.
//
// Rules, per window:
//
//  * Lines. For each payline and each paying symbol `s`, the run length `k` is the number of
//    leading cells, from reel 1, showing `s` or the wild. The line pays `linePay[s][k]`. Only
//    the best symbol on a line pays: highest pay, and on a tie the longer run. Its `k` leading
//    cells are the line's winning cells.
//  * Cascades. Winning cells clear, survivors drop, and each reel refills from its own strip:
//    clearing `d` cells moves the reel's offset back by `d` and brings in the `d` symbols above
//    the old window. Repeat until a window pays nothing. Cascade step `j` (the first win is
//    step 0) is multiplied by `ladder[min(j, ladder.length - 1)]`.
//  * Scatter. Scatters are never part of a line, so they never clear. Once the chain has ended,
//    the visible scatters on the final window are counted and pay `scatterPay[count]`, a
//    multiple of the stake, unmultiplied, once per spin.

import gameJson from './game.json'

export interface GameParameters {
  /** Symbol names; a cell holds an index into this. */
  symbols: string[]
  wild: number
  scatter: number
  /** Indices of the symbols that carry a line pay. */
  paying: number[]
  reels: number
  rows: number
  length: number
  /** `lines[i][reel]` is the row that payline `i` occupies on that reel. */
  lines: number[][]
  /** `linePay[symbol][runLength]`, in units. */
  linePay: number[][]
  /** `scatterPay[visibleCount]`, in units. */
  scatterPay: number[]
  ladder: number[]
  maxSteps: number
  /** What one spin costs: 20 paylines of 5 units. */
  stakeUnits: number
  weightTotal: number
}

export const GAME = gameJson as GameParameters

/** One payline's win within a window. */
export interface LineWin {
  line: number
  symbol: number
  runLength: number
  units: number
}

/** One window of a spin: what it showed, what it paid, and which cells cleared. */
export interface CascadeStep {
  /** `grid[reel][row]`, symbol indices, as the window stood when it was evaluated. */
  grid: number[][]
  /** Where each reel's strip sat when this window was built. */
  offsets: number[]
  /**
   * `origins[reel][row]` is the stop each visible cell came from. After a cascade a window is
   * no longer three consecutive stops: the cells that cleared refill from the strip while the
   * survivors keep whatever they were, so only this says truthfully where the window sits.
   */
  origins: number[][]
  wins: LineWin[]
  /** The window's line pay before the ladder, in units. */
  lineUnits: number
  multiplier: number
  /** `lineUnits * multiplier`. */
  units: number
  /** Winning cells, as a bitmask over `reel * rows + row`. */
  cleared: number
}

export interface SpinResult {
  /** Total pay in units, scatter included. */
  totalUnits: number
  /** What the first window paid, after its ×1 multiplier. */
  baseUnits: number
  /** Windows that paid a line win: 0 when the lines paid nothing. */
  steps: number
  /** A chain that hit `maxSteps`; never seen in this game. */
  runaway: boolean
  scatterCount: number
  scatterUnits: number
  /** The window the chain ended on — only populated when tracing. */
  finalGrid: number[][] | null
  /** Where the strips sat on that final window — only populated when tracing. */
  finalOffsets: number[] | null
  /** The stop each cell of that final window came from — only populated when tracing. */
  finalOrigins: number[][] | null
}

export const cellBit = (reel: number, row: number, rows = GAME.rows): number =>
  1 << (reel * rows + row)

/** Symbol names to indices, for the strips as `designs.json` stores them. */
export function stripsToIndices(strips: readonly (readonly string[])[], game = GAME): number[][] {
  const index = new Map(game.symbols.map((s, i) => [s, i]))
  return strips.map((strip) =>
    strip.map((name) => {
      const i = index.get(name)
      if (i === undefined) throw new Error(`unknown symbol ${name}`)
      return i
    }),
  )
}

/**
 * One evaluator with its own scratch window, so a long run allocates nothing per spin beyond
 * its result. The Python simulator owns its scratch the same way.
 */
export class Evaluator {
  private readonly grid: number[][]
  private readonly origins: number[][]
  private readonly offsets: number[]
  private readonly gone: number[]
  private readonly keep: number[]
  private readonly keepFrom: number[]

  constructor(
    private readonly strips: readonly (readonly number[])[],
    private readonly game: GameParameters = GAME,
  ) {
    const { reels, rows, length } = game
    if (strips.length !== reels) throw new Error(`expected ${reels} strips`)
    for (const strip of strips) {
      if (strip.length !== length) throw new Error(`every strip must be ${length} long`)
    }
    this.grid = Array.from({ length: reels }, () => new Array<number>(rows).fill(0))
    this.origins = Array.from({ length: reels }, () => new Array<number>(rows).fill(0))
    this.offsets = new Array<number>(reels).fill(0)
    this.gone = new Array<number>(rows).fill(0)
    this.keep = new Array<number>(rows).fill(0)
    this.keepFrom = new Array<number>(rows).fill(0)
  }

  /**
   * Line pay of one window in units, and its winning cells as a bitmask. When `wins` is given,
   * each paying line is appended to it.
   */
  private evaluateWindow(wins: LineWin[] | null): { units: number; cleared: number } {
    const { reels, rows, wild, paying, lines, linePay } = this.game
    const grid = this.grid
    let units = 0
    let cleared = 0
    for (let li = 0; li < lines.length; li++) {
      const line = lines[li]!
      let bestPay = 0
      let bestRun = 0
      let bestSymbol = -1
      for (const symbol of paying) {
        let k = 0
        while (k < reels) {
          const cell = grid[k]![line[k]!]!
          if (cell !== symbol && cell !== wild) break
          k++
        }
        const pay = linePay[symbol]![k]!
        // Highest pay wins; on a tie, the longer run. Strictly greater, so the first symbol
        // in alphabet order keeps a tie, as `reference.py` does.
        if (pay > 0 && (pay > bestPay || (pay === bestPay && k > bestRun))) {
          bestPay = pay
          bestRun = k
          bestSymbol = symbol
        }
      }
      if (bestPay > 0) {
        units += bestPay
        for (let r = 0; r < bestRun; r++) cleared |= 1 << (r * rows + line[r]!)
        if (wins) wins.push({ line: li, symbol: bestSymbol, runLength: bestRun, units: bestPay })
      }
    }
    return { units, cleared }
  }

  /**
   * One spin from the given stops. Pass `trace` to record each window for the animation; leave
   * it null and the spin allocates nothing it does not return.
   */
  play(stops: ArrayLike<number>, trace: CascadeStep[] | null = null): SpinResult {
    const { reels, rows, length, ladder, maxSteps, scatter, scatterPay } = this.game
    const { grid, origins, offsets, gone, keep, keepFrom } = this
    for (let r = 0; r < reels; r++) {
      offsets[r] = stops[r]!
      const strip = this.strips[r]!
      for (let k = 0; k < rows; k++) {
        const stop = (offsets[r]! + k) % length
        grid[r]![k] = strip[stop]!
        origins[r]![k] = stop
      }
    }

    let totalUnits = 0
    let baseUnits = 0
    let steps = 0
    let runaway = false
    for (;;) {
      const wins: LineWin[] | null = trace ? [] : null
      const { units: lineUnits, cleared } = this.evaluateWindow(wins)
      if (lineUnits === 0) break
      const multiplier = ladder[Math.min(steps, ladder.length - 1)]!
      const units = lineUnits * multiplier
      totalUnits += units
      if (steps === 0) baseUnits = units
      steps++
      if (trace) {
        trace.push({
          grid: grid.map((column) => column.slice()),
          offsets: offsets.slice(),
          origins: origins.map((column) => column.slice()),
          wins: wins!,
          lineUnits,
          multiplier,
          units,
          cleared,
        })
      }
      if (steps >= maxSteps) {
        runaway = true
        break
      }
      for (let r = 0; r < reels; r++) {
        let goneCount = 0
        let keepCount = 0
        for (let k = 0; k < rows; k++) {
          if (cleared & (1 << (r * rows + k))) gone[goneCount++] = k
          else {
            keep[keepCount] = grid[r]![k]!
            keepFrom[keepCount] = origins[r]![k]!
            keepCount++
          }
        }
        if (goneCount === 0) continue
        offsets[r] = (offsets[r]! - goneCount + length) % length
        const strip = this.strips[r]!
        for (let j = 0; j < goneCount; j++) {
          const stop = (offsets[r]! + j) % length
          grid[r]![j] = strip[stop]!
          origins[r]![j] = stop
        }
        for (let j = 0; j < keepCount; j++) {
          grid[r]![goneCount + j] = keep[j]!
          origins[r]![goneCount + j] = keepFrom[j]!
        }
      }
    }

    let scatterCount = 0
    for (let r = 0; r < reels; r++) {
      for (let k = 0; k < rows; k++) if (grid[r]![k] === scatter) scatterCount++
    }
    const scatterUnits = scatterPay[Math.min(scatterCount, scatterPay.length - 1)]!
    return {
      totalUnits: totalUnits + scatterUnits,
      baseUnits,
      steps,
      runaway,
      scatterCount,
      scatterUnits,
      finalGrid: trace ? grid.map((column) => column.slice()) : null,
      finalOffsets: trace ? offsets.slice() : null,
      finalOrigins: trace ? origins.map((column) => column.slice()) : null,
    }
  }
}
