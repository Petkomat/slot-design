// Serves the fitted design to the page as a virtual module, read straight from the fitter's
// own output. The page therefore cannot drift from `slotmath/designs.json`: a refit changes
// what the browser plays, and the agreement fixture's digest turns a stale check into a
// failure. Only the fields the page needs are shipped; the solver's working notes are not.

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Plugin } from 'vite'
import type { FittedDesign } from './src/design.ts'

export type { FittedDesign }

const HERE = dirname(fileURLToPath(import.meta.url))
export const DESIGNS_FILE = resolve(HERE, '..', 'slotmath', 'designs.json')

const VIRTUAL_ID = 'virtual:fitted-design'
const RESOLVED_ID = '\0' + VIRTUAL_ID

export function readFittedDesign(): FittedDesign {
  const all = JSON.parse(readFileSync(DESIGNS_FILE, 'utf8'))
  const key: string = all.main
  const d = all.games[key]
  return {
    key,
    strips: d.strips,
    weights: d.weights,
    targetRtp: all.inputs.rtp,
    targetVolatility: d.target_volatility,
    exactRtp: d.rtp,
    exactVolatility: d.volatility,
    exactHitFrequency: d.hit_frequency,
    maxWin: d.max_win,
  }
}

export function fittedDesignPlugin(): Plugin {
  return {
    name: 'fitted-design',
    resolveId: (id) => (id === VIRTUAL_ID ? RESOLVED_ID : null),
    load(id) {
      if (id !== RESOLVED_ID) return null
      this.addWatchFile(DESIGNS_FILE)
      return `export default ${JSON.stringify(readFittedDesign())}`
    },
  }
}
