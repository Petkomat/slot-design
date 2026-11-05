// The slow tier: `npm run test:slow`, or the "tests" workflow dispatched with "slow" ticked.
//
// A million spins of the standard stream through the TypeScript evaluator, against the totals
// `slotmath/webfixture.py` recorded from the Python ones. The comparison is exact. Checking an
// empirical RTP against 0.96 instead would have a standard error of 0.0045 at this many spins:
// blind to anything smaller than about 1% of RTP, and failing by chance one run in twenty.

import { describe, expect, it } from 'vitest'
import { expectWithin53Bits, fixture, runStream } from './harness'

describe('one million spins', () => {
  it('matches reference.py exactly', () => {
    const run = fixture.runs.slow
    const got = runStream(run.spins, run.seed, run.stream)
    expectWithin53Bits(got)
    expect(got.pay_sum).toBe(run.pay_sum)
    expect(got.pay_sq).toBe(run.pay_sq)
    expect(got.hits).toBe(run.hits)
    expect(got.steps).toEqual(run.steps)
  })
})
