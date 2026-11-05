// Shared setup for the agreement tests: the fitted design, the evaluator, and a run of the
// standard seeded stream reduced to the same totals `slotmath/webfixture.py` records.

import { createHash } from 'node:crypto'
import design from 'virtual:fitted-design'
import fixture from '../src/fixture.json'
import { StopSampler, lookupTables } from '../src/rng'
import { Evaluator, GAME, stripsToIndices } from '../src/rules'

export { design, fixture }

/** The same hash `webfixture.design_digest` takes, over the same compact JSON. */
export function designDigest(): string {
  const payload = JSON.stringify([design.strips, design.weights])
  return createHash('sha256').update(payload).digest('hex').slice(0, 16)
}

export interface Totals {
  spins: number
  pay_sum: number
  pay_sq: number
  hits: number
  steps: number[]
}

/** `spins` spins of the standard stream, reduced to the fixture's totals. */
export function runStream(spins: number, seed: number, stream: number): Totals {
  const evaluator = new Evaluator(stripsToIndices(design.strips))
  const sampler = new StopSampler(lookupTables(design.weights, GAME.weightTotal), seed, stream)
  const stops = new Uint8Array(GAME.reels)
  const steps: number[] = []
  let paySum = 0
  let paySq = 0
  let hits = 0
  for (let i = 0; i < spins; i++) {
    sampler.next(stops)
    const result = evaluator.play(stops)
    paySum += result.totalUnits
    paySq += result.totalUnits * result.totalUnits
    if (result.totalUnits > 0) hits++
    steps[result.steps] = (steps[result.steps] ?? 0) + 1
  }
  for (let i = 0; i < steps.length; i++) steps[i] ??= 0
  return { spins, pay_sum: paySum, pay_sq: paySq, hits, steps }
}

/** Totals are exact integers well inside 2**53, so they compare with `toBe`, not a tolerance. */
export function expectWithin53Bits(totals: Totals): void {
  for (const value of [totals.pay_sum, totals.pay_sq]) {
    if (!Number.isSafeInteger(value)) throw new Error(`${value} has lost integer precision`)
  }
}
