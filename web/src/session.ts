// What the player has actually got so far, and how uncertain it is.
//
// The panel shows three readings of each figure and never conflates them: the *target* the
// fitter was asked for, the *exact* value the fitted integer weights achieve, and the *session*
// value, which is a random variable. Session figures therefore carry a standard error. After a
// hundred spins the session RTP's standard error is 4.5 / sqrt(100) = 0.45, so a bare 0.62 on
// screen would make a correct fit look broken; 0.62 +/- 0.45 does not.

/** Raw moments of the pay in units, enough for RTP, volatility and both standard errors. */
export interface Totals {
  spins: number
  s1: number
  s2: number
  s3: number
  s4: number
  hits: number
}

export const emptyTotals = (): Totals => ({ spins: 0, s1: 0, s2: 0, s3: 0, s4: 0, hits: 0 })

export function addSpin(totals: Totals, payUnits: number): void {
  const x2 = payUnits * payUnits
  totals.spins++
  totals.s1 += payUnits
  totals.s2 += x2
  totals.s3 += x2 * payUnits
  totals.s4 += x2 * x2
  if (payUnits > 0) totals.hits++
}

export function mergeTotals(a: Totals, b: Totals): Totals {
  return {
    spins: a.spins + b.spins,
    s1: a.s1 + b.s1,
    s2: a.s2 + b.s2,
    s3: a.s3 + b.s3,
    s4: a.s4 + b.s4,
    hits: a.hits + b.hits,
  }
}

export interface Summary {
  spins: number
  /** Per unit staked. */
  rtp: number
  rtpSe: number
  /** Standard deviation of one spin's pay per unit staked. */
  volatility: number
  volatilitySe: number
  hitFrequency: number
  /** Returned minus staked, in stakes. */
  net: number
}

/**
 * The session figures with their standard errors.
 *
 * RTP's is the textbook sigma / sqrt(n). The volatility's needs the fourth moment: the sample
 * variance has variance (mu4 - sigma^4) / n, and the standard deviation is its square root, so
 * its standard error is sqrt((mu4 - sigma^4) / n) / (2 * sigma). With a 500x top prize the
 * fourth moment is dominated by a handful of spins, so this band is wide and jumps about. That
 * is the honest picture, and the reason volatility is the slower of the two to pin down.
 */
export function summarise(totals: Totals, stakeUnits: number): Summary {
  const { spins, s1, s2, s3, s4, hits } = totals
  if (spins === 0) {
    return {
      spins: 0,
      rtp: Number.NaN,
      rtpSe: Number.NaN,
      volatility: Number.NaN,
      volatilitySe: Number.NaN,
      hitFrequency: Number.NaN,
      net: 0,
    }
  }
  const m1 = s1 / spins
  const m2 = s2 / spins
  const m3 = s3 / spins
  const m4 = s4 / spins
  // Unbiased, so a single spin reports no spread rather than a spurious zero: one spin carries
  // no information about how much payouts vary, and "4.0000 +/- 0.0000" would claim it does.
  const variance = spins > 1 ? (Math.max(m2 - m1 * m1, 0) * spins) / (spins - 1) : Number.NaN
  const sd = Math.sqrt(variance) // undefined for a single spin, and stays so downstream
  // Fourth central moment from the raw ones.
  const mu4 = Math.max(m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 ** 4, 0)
  const varianceOfVariance = Math.max(mu4 - variance * variance, 0) / spins
  const sdSe = sd > 0 ? Math.sqrt(varianceOfVariance) / (2 * sd) : Number.NaN
  return {
    spins,
    rtp: m1 / stakeUnits,
    rtpSe: sd / Math.sqrt(spins) / stakeUnits,
    volatility: sd / stakeUnits,
    volatilitySe: sdSe / stakeUnits,
    hitFrequency: hits / spins,
    net: (s1 - spins * stakeUnits) / stakeUnits,
  }
}
