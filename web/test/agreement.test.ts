// The fast tier, run on every push. The million-spin run is in agreement.slow.test.ts.
//
// Each layer is pinned separately so a failure says where the fault is: the design the page
// loaded, the random stream, or the evaluator reading it.

import { describe, expect, it } from 'vitest'
import { Splitmix64, StopSampler, lookupTables } from '../src/rng'
import { type CascadeStep, Evaluator, GAME, stripsToIndices } from '../src/rules'
import { design, designDigest, expectWithin53Bits, fixture, runStream } from './harness'

describe('the design the page plays', () => {
  it('is the one the fixture was recorded from', () => {
    // A refit changes designs.json; without regenerating the fixture the totals below would be
    // compared against a different game. This turns that into a named failure.
    expect(design.key).toBe(fixture.design)
    expect(designDigest()).toBe(fixture.digest)
  })

  it('carries weights summing to 4096 on every reel', () => {
    for (const reel of design.weights) {
      expect(reel.reduce((a, b) => a + b, 0)).toBe(GAME.weightTotal)
    }
  })

  it('agrees with the fitter on stake and geometry', () => {
    expect(GAME.stakeUnits).toBe(fixture.stake_units)
    expect(design.strips).toHaveLength(GAME.reels)
    expect(design.strips[0]).toHaveLength(GAME.length)
    expect(GAME.lines).toHaveLength(20)
  })
})

describe('the random stream', () => {
  it('reproduces the recorded splitmix64 outputs', () => {
    const rng = new Splitmix64(fixture.sample.seed, fixture.sample.stream)
    const got = fixture.sample.draws.map(() => rng.next().toString())
    expect(got).toEqual(fixture.sample.draws)
  })

  it('reproduces the recorded stop vectors', () => {
    const tables = lookupTables(design.weights, GAME.weightTotal)
    const sampler = new StopSampler(tables, fixture.sample.seed, fixture.sample.stream)
    const stops = new Uint8Array(GAME.reels)
    const got = fixture.sample.stops.map(() => Array.from(sampler.next(stops)))
    expect(got).toEqual(fixture.sample.stops)
  })

  it('builds lookup tables in which a stop appears exactly its weight times', () => {
    const [table] = lookupTables(design.weights, GAME.weightTotal)
    const counts = new Array<number>(GAME.length).fill(0)
    for (const stop of table!) counts[stop]!++
    expect(counts).toEqual(design.weights[0])
  })
})

describe('the evaluator', () => {
  it('matches reference.py on 20,000 spins of the standard stream, exactly', () => {
    const run = fixture.runs.fast
    const got = runStream(run.spins, run.seed, run.stream)
    expectWithin53Bits(got)
    expect(got.pay_sum).toBe(run.pay_sum)
    expect(got.pay_sq).toBe(run.pay_sq)
    expect(got.hits).toBe(run.hits)
    expect(got.steps).toEqual(run.steps)
  })

  it('pays nothing when no symbol reaches three reels', () => {
    // One symbol per reel, so every payline runs one long whichever rows it takes.
    const evaluator = new Evaluator(constantReels(['L1', 'L2', 'L3', 'L4', 'L5'], 3), {
      ...GAME,
      length: 3,
    })
    const result = evaluator.play([0, 0, 0, 0, 0])
    expect(result.totalUnits).toBe(0)
    expect(result.steps).toBe(0)
    expect(result.scatterCount).toBe(0)
  })

  it('treats the wild as every paying symbol, and clears only the winning run', () => {
    // H1 W H1 on reels 1-3 runs three long on all 20 lines; reels 4-5 end it. The three
    // winning reels refill with a symbol apiece, so the chain stops after one window.
    const strips = stripsToIndices([
      ['H1', 'H1', 'H1', 'L1', 'L1', 'L1'],
      ['W', 'W', 'W', 'L2', 'L2', 'L2'],
      ['H1', 'H1', 'H1', 'L3', 'L3', 'L3'],
      ['L4', 'L4', 'L4', 'L4', 'L4', 'L4'],
      ['L5', 'L5', 'L5', 'L5', 'L5', 'L5'],
    ])
    const evaluator = new Evaluator(strips, { ...GAME, length: 6 })
    const trace: CascadeStep[] = []
    const result = evaluator.play([0, 0, 0, 0, 0], trace)

    expect(trace).toHaveLength(1)
    expect(trace[0]!.wins).toHaveLength(GAME.lines.length)
    for (const win of trace[0]!.wins) {
      expect(GAME.symbols[win.symbol]).toBe('H1')
      expect(win.runLength).toBe(3)
      expect(win.units).toBe(GAME.linePay[0]![3])
    }
    // Reels 1-3 clear entirely; reels 4-5 are untouched.
    expect(trace[0]!.cleared).toBe(0b000000111111111)
    expect(result.steps).toBe(1)
    expect(result.totalUnits).toBe(GAME.lines.length * GAME.linePay[0]![3]!)
  })

  it('climbs the multiplier ladder across a cascade', () => {
    // H1 on all three reels, refilling to H2, then to a dead window. Step 0 pays x1 and
    // step 1 pays x2, so the ladder is visible in the total.
    const strips = stripsToIndices([
      ['H1', 'H1', 'H1', 'L1', 'L1', 'L1', 'H2', 'H2', 'H2'],
      ['H1', 'H1', 'H1', 'L2', 'L2', 'L2', 'H2', 'H2', 'H2'],
      ['H1', 'H1', 'H1', 'L3', 'L3', 'L3', 'H2', 'H2', 'H2'],
      ['L4', 'L4', 'L4', 'L4', 'L4', 'L4', 'L4', 'L4', 'L4'],
      ['L5', 'L5', 'L5', 'L5', 'L5', 'L5', 'L5', 'L5', 'L5'],
    ])
    const evaluator = new Evaluator(strips, { ...GAME, length: 9 })
    const trace: CascadeStep[] = []
    const result = evaluator.play([0, 0, 0, 0, 0], trace)

    const lines = GAME.lines.length
    const h1 = lines * GAME.linePay[0]![3]!
    const h2 = lines * GAME.linePay[1]![3]!
    expect(trace.map((step) => step.multiplier)).toEqual([GAME.ladder[0], GAME.ladder[1]])
    expect(trace.map((step) => step.units)).toEqual([h1 * GAME.ladder[0]!, h2 * GAME.ladder[1]!])
    expect(result.steps).toBe(2)
    expect(result.baseUnits).toBe(h1 * GAME.ladder[0]!)
    expect(result.totalUnits).toBe(h1 * GAME.ladder[0]! + h2 * GAME.ladder[1]!)
  })
})

describe('where the window sits on the strips', () => {
  it('names the stop every visible cell came from, cascade or not', () => {
    // The page outlines these stops on the strips panel. After a cascade they are no longer
    // three consecutive stops, so the claim is only as good as this check.
    const stripIndices = stripsToIndices(design.strips)
    const evaluator = new Evaluator(stripIndices)
    const sampler = new StopSampler(
      lookupTables(design.weights, GAME.weightTotal),
      fixture.sample.seed,
      fixture.sample.stream,
    )
    const stops = new Uint8Array(GAME.reels)
    let cascaded = 0

    for (let spin = 0; spin < 2000; spin++) {
      sampler.next(stops)
      const trace: CascadeStep[] = []
      const result = evaluator.play(stops, trace)
      if (result.steps > 1) cascaded++
      const windows = [...trace.map((s) => ({ grid: s.grid, origins: s.origins }))]
      windows.push({ grid: result.finalGrid!, origins: result.finalOrigins! })
      for (const { grid, origins } of windows) {
        for (let r = 0; r < GAME.reels; r++) {
          for (let k = 0; k < GAME.rows; k++) {
            expect(stripIndices[r]![origins[r]![k]!]).toBe(grid[r]![k])
          }
        }
      }
    }
    // Without cascades in the sample the check would prove only the easy case.
    expect(cascaded).toBeGreaterThan(50)
  })
})

/** Five reels, each showing one symbol everywhere: a window with nothing accidental in it. */
function constantReels(symbols: string[], length: number): number[][] {
  return stripsToIndices(symbols.map((s) => new Array<string>(length).fill(s)))
}
