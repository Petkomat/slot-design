// splitmix64 and the stop sampler, reproducing `slotmath/simulate.py` exactly.
//
// The Python side seeds a stream with `mix(mix(seed + GOLDEN * (stream + 1)))`, then advances
// `state += GOLDEN` and returns `mix(state)`. Each 64-bit draw carries five 12-bit indices into
// per-reel lookup tables in which stop `t` appears `weight[t]` times, so sampling is one table
// read and has no modulo bias. Five reels fit in one draw.
//
// Because the arithmetic matches, any spin shown in the browser can be replayed in Python:
// `simulate_python(game, n, seed=<the seed on screen>)` visits the same stops in the same order.

const MASK = (1n << 64n) - 1n
const GOLDEN = 0x9e3779b97f4a7c15n
const M1 = 0xbf58476d1ce4e5b9n
const M2 = 0x94d049bb133111ebn
const LOW12 = 0xfffn

function mix(z: bigint): bigint {
  z = ((z ^ (z >> 30n)) * M1) & MASK
  z = ((z ^ (z >> 27n)) * M2) & MASK
  return (z ^ (z >> 31n)) & MASK
}

/** The initial state of one stream, as `stream_state_python` derives it: two mixes. */
export function streamState(seed: number | bigint, stream: number | bigint): bigint {
  return mix(mix((BigInt(seed) + GOLDEN * (BigInt(stream) + 1n)) & MASK))
}

/** The 64-bit generator. One instance is one stream. */
export class Splitmix64 {
  private state: bigint

  constructor(seed: number | bigint, stream: number | bigint = 0) {
    this.state = streamState(seed, stream)
  }

  /** The next 64-bit output. */
  next(): bigint {
    this.state = (this.state + GOLDEN) & MASK
    return mix(this.state)
  }

  /**
   * The raw state, so a run can be handed to a worker and continued there rather than
   * restarted. Moving the state keeps one unbroken stream from the seed on screen.
   */
  get position(): bigint {
    return this.state
  }

  set position(state: bigint) {
    this.state = state & MASK
  }
}

/**
 * Per reel, 4096 entries mapping 12 random bits to a stop, with stop `t` appearing `weight[t]`
 * times. Weights must sum to 4096 on every reel, as the fitter guarantees.
 */
export function lookupTables(weights: readonly (readonly number[])[], total = 4096): Uint8Array[] {
  return weights.map((row, reel) => {
    const table = new Uint8Array(total)
    let at = 0
    for (let stop = 0; stop < row.length; stop++) {
      for (let k = 0; k < row[stop]!; k++) table[at++] = stop
    }
    if (at !== total) throw new Error(`reel ${reel + 1}: weights sum to ${at}, not ${total}`)
    return table
  })
}

/** Draws one stop per reel into `stops`, in the order `simulate.py` draws them. */
export class StopSampler {
  private readonly rng: Splitmix64
  private draw = 0n
  private used = 0

  constructor(
    private readonly tables: readonly Uint8Array[],
    seed: number | bigint,
    stream: number | bigint = 0,
  ) {
    this.rng = new Splitmix64(seed, stream)
  }

  /** Where the stream has reached; hand it to a worker to continue the same sequence. */
  get position(): bigint {
    return this.rng.position
  }

  set position(state: bigint) {
    this.rng.position = state
  }

  /**
   * One spin's stops. The Python loop takes a fresh draw at the start of every spin and then
   * consumes five 12-bit fields from it, so with five reels each spin is exactly one draw.
   */
  next(stops: Uint8Array): Uint8Array {
    this.draw = this.rng.next()
    this.used = 0
    for (let reel = 0; reel < this.tables.length; reel++) {
      if (this.used === 5) {
        this.draw = this.rng.next()
        this.used = 0
      }
      const bits = Number((this.draw >> BigInt(12 * this.used)) & LOW12)
      stops[reel] = this.tables[reel]![bits]!
      this.used++
    }
    return stops
  }
}
