// The shape of the fitted design the page is handed. Defined here so that the build-time
// reader (`vite-plugin-design.ts`) and the module declaration for `virtual:fitted-design`
// cannot drift apart.

export interface FittedDesign {
  /** Which fitted level this is: "v3" is the main design. */
  key: string
  /** Reel strips: `strips[reel][stop]` is a symbol name. */
  strips: string[][]
  /** Stop weights, summing to 4096 on every reel. */
  weights: number[][]
  /** What the fitter was asked for. */
  targetRtp: number
  targetVolatility: number
  /** What the fitted integer weights achieve, by exact enumeration. */
  exactRtp: number
  exactVolatility: number
  exactHitFrequency: number
  /** The largest possible win, in stakes. */
  maxWin: number
}
