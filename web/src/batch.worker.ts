// Batch spinning, off the main thread.
//
// A million spins takes a second or two, and the whole point of the batch button is to watch
// the session figures converge while it runs — which cannot happen if the tab is blocked. The
// worker is handed the stream's current position rather than a seed, so a batch continues the
// same sequence the hand-played spins were drawing from, and hands the position back.

import design from 'virtual:fitted-design'
import { StopSampler, lookupTables } from './rng'
import { Evaluator, GAME, stripsToIndices } from './rules'
import { type Totals, addSpin, emptyTotals } from './session'

export interface BatchRequest {
  spins: number
  seed: number
  stream: number
  position: bigint
}

export type BatchMessage =
  | { type: 'progress'; totals: Totals }
  | { type: 'done'; totals: Totals; position: bigint }

const scope = self as unknown as Worker
const evaluator = new Evaluator(stripsToIndices(design.strips))
const tables = lookupTables(design.weights, GAME.weightTotal)

scope.addEventListener('message', (event) => {
  const { spins, seed, stream, position } = (event as MessageEvent<BatchRequest>).data
  const sampler = new StopSampler(tables, seed, stream)
  sampler.position = position
  const stops = new Uint8Array(GAME.reels)
  const totals = emptyTotals()
  // About forty updates over the run: often enough to look continuous, rare enough to cost
  // nothing. Posting every spin would be slower than the spins themselves.
  const report = Math.max(1, Math.floor(spins / 40))

  for (let i = 0; i < spins; i++) {
    sampler.next(stops)
    addSpin(totals, evaluator.play(stops).totalUnits)
    if ((i + 1) % report === 0 && i + 1 < spins) {
      scope.postMessage({ type: 'progress', totals: { ...totals } } satisfies BatchMessage)
    }
  }
  scope.postMessage({
    type: 'done',
    totals,
    position: sampler.position,
  } satisfies BatchMessage)
})
