// The page: one fitted design, played on the fitted strips, with the session's figures kept
// next to the ones the fitter was asked for and the ones it achieved.

import design from 'virtual:fitted-design'
import type { BatchMessage, BatchRequest } from './batch.worker'
import { StopSampler, lookupTables } from './rng'
import { type CascadeStep, Evaluator, GAME, stripsToIndices } from './rules'
import { type Totals, addSpin, emptyTotals, mergeTotals, summarise } from './session'
import { spriteMarkup, symbolMarkup } from './symbols'
import './style.css'

const STREAM = 0
const SHOW_MS = 420 // a new window, settling
const WIN_MS = 820 // winning cells lit
const CLEAR_MS = 320 // winning cells leaving

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id)
  if (!node) throw new Error(`missing element #${id}`)
  return node as T
}

const ui = {
  sprite: el('sprite'),
  reels: el('reels'),
  ladder: el('ladder'),
  spinTotal: el('spin-total'),
  winList: el<HTMLOListElement>('win-list'),
  winsEmpty: el('wins-empty'),
  strips: el('strips'),
  progress: el('progress'),
  spin: el<HTMLButtonElement>('spin'),
  batch10k: el<HTMLButtonElement>('batch-10k'),
  batch1m: el<HTMLButtonElement>('batch-1m'),
  reset: el<HTMLButtonElement>('reset'),
  seedInput: el<HTMLInputElement>('seed-input'),
  targetRtp: el('target-rtp'),
  targetVol: el('target-vol'),
  exactRtp: el('exact-rtp'),
  exactVol: el('exact-vol'),
  sessionRtp: el('session-rtp'),
  sessionVol: el('session-vol'),
  countSpins: el('count-spins'),
  countHits: el('count-hits'),
  countNet: el('count-net'),
}

// ----------------------------------------------------------------- the game
const strips = stripsToIndices(design.strips)
const tables = lookupTables(design.weights, GAME.weightTotal)
const evaluator = new Evaluator(strips)

let seed = Number(ui.seedInput.value)
let sampler = new StopSampler(tables, seed, STREAM)
let committed: Totals = emptyTotals()
let pending: Totals | null = null
let busy = false
let skip = false

const worker = new Worker(new URL('./batch.worker.ts', import.meta.url), { type: 'module' })

// ------------------------------------------------------------- number formats
const fixed = (value: number, places: number): string =>
  Number.isFinite(value) ? value.toFixed(places) : '—'

const plusMinus = (value: number, error: number, places: number): string =>
  Number.isFinite(value) ? `${value.toFixed(places)} ± ${fixed(error, places)}` : '—'

const stakes = (units: number): string => `${(units / GAME.stakeUnits).toFixed(2)}×`

// --------------------------------------------------------------- the figures
function renderFigures(): void {
  ui.targetRtp.textContent = design.targetRtp.toFixed(4)
  ui.targetVol.textContent = design.targetVolatility.toFixed(3)
  ui.exactRtp.textContent = design.exactRtp.toFixed(10)
  ui.exactVol.textContent = design.exactVolatility.toFixed(6)

  const totals = pending ? mergeTotals(committed, pending) : committed
  const s = summarise(totals, GAME.stakeUnits)
  if (s.spins === 0) {
    ui.sessionRtp.textContent = '—'
    ui.sessionVol.textContent = '—'
    ui.countSpins.textContent = '0'
    ui.countHits.textContent = '—'
    ui.countNet.textContent = '0.00×'
    return
  }
  ui.sessionRtp.textContent = plusMinus(s.rtp, s.rtpSe, 4)
  ui.sessionVol.textContent = plusMinus(s.volatility, s.volatilitySe, 3)
  ui.countSpins.textContent = s.spins.toLocaleString('en-GB')
  ui.countHits.textContent = `${(s.hitFrequency * 100).toFixed(2)}%`
  ui.countNet.textContent = `${s.net >= 0 ? '+' : '−'}${Math.abs(s.net).toFixed(2)}×`
  ui.countNet.classList.toggle('up', s.net > 0)
  ui.countNet.classList.toggle('down', s.net < 0)
}

// ---------------------------------------------------------------- the window
/** The cells a refill brings in: the top `d` rows of every reel that cleared `d` cells. */
function refillMask(cleared: number): number {
  let mask = 0
  for (let r = 0; r < GAME.reels; r++) {
    let dropped = 0
    for (let k = 0; k < GAME.rows; k++) if (cleared & (1 << (r * GAME.rows + k))) dropped++
    for (let k = 0; k < dropped; k++) mask |= 1 << (r * GAME.rows + k)
  }
  return mask
}

interface WindowFlags {
  winning?: number
  clearing?: number
  dropping?: number
}

function renderWindow(grid: number[][], flags: WindowFlags = {}): void {
  const cells: string[] = []
  for (let k = 0; k < GAME.rows; k++) {
    for (let r = 0; r < GAME.reels; r++) {
      const bit = 1 << (r * GAME.rows + k)
      const classes = ['cell']
      if (flags.winning && flags.winning & bit) classes.push('is-winning')
      if (flags.clearing && flags.clearing & bit) classes.push('is-clearing')
      if (flags.dropping && flags.dropping & bit) classes.push('is-dropping')
      const name = GAME.symbols[grid[r]![k]!]!
      cells.push(
        `<div class="${classes.join(' ')}" role="gridcell" style="--drop-delay:${k * 60}ms">` +
          `${symbolMarkup(name)}</div>`,
      )
    }
  }
  ui.reels.innerHTML = cells.join('')
}

function renderLadder(step: number | null): void {
  ui.ladder.innerHTML = GAME.ladder
    .map((m, i) => {
      const last = i === GAME.ladder.length - 1
      const active = step !== null && (step === i || (last && step > i))
      return `<span class="rung${active ? ' is-active' : ''}">×${m}</span>`
    })
    .join('')
}

// -------------------------------------------------------------- the win list
function clearWins(): void {
  ui.winList.innerHTML = ''
  ui.winsEmpty.hidden = false
  ui.spinTotal.innerHTML = '&nbsp;'
}

function appendStepWins(step: CascadeStep, index: number): void {
  ui.winsEmpty.hidden = true
  const items = step.wins
    .slice()
    .sort((a, b) => b.units - a.units)
    .map((win) => {
      const symbol = GAME.symbols[win.symbol]!
      return (
        `<li class="win"><span class="win-symbol">${symbolMarkup(symbol, false)}</span>` +
        `<span class="win-text"><strong>${symbol}</strong> ×${win.runLength} ` +
        `on line ${win.line + 1}</span>` +
        `<span class="win-units">${stakes(win.units * step.multiplier)}</span></li>`
      )
    })
  const heading =
    `<li class="win-step"><span>Cascade step ${index}</span>` +
    `<span class="win-step-mult">×${step.multiplier}</span>` +
    `<span class="win-units">${stakes(step.units)}</span></li>`
  ui.winList.insertAdjacentHTML('beforeend', heading + items.join(''))
}

// --------------------------------------------------------------- the strips
const stripCells: HTMLElement[][] = []

function buildStrips(): void {
  const columns = design.strips.map((strip, r) => {
    const cells = strip
      .map((name, stop) => {
        const weight = design.weights[r]![stop]!
        const share = ((weight / GAME.weightTotal) * 100).toFixed(2)
        return (
          `<div class="strip-cell" data-reel="${r}" data-stop="${stop}" ` +
          `title="reel ${r + 1}, stop ${stop}: ${name}, weight ${weight}/${GAME.weightTotal} ` +
          `= ${share}%" tabindex="0">${symbolMarkup(name, false)}` +
          `<span class="strip-weight">${weight}</span></div>`
        )
      })
      .join('')
    return `<div class="strip"><h3>Reel ${r + 1}</h3><div class="strip-cells">${cells}</div></div>`
  })
  ui.strips.innerHTML = columns.join('')
  for (let r = 0; r < GAME.reels; r++) {
    stripCells[r] = Array.from(
      ui.strips.querySelectorAll<HTMLElement>(`.strip-cell[data-reel="${r}"]`),
    )
  }
}

/**
 * Outlines the stops the window is actually showing. After a cascade these are not three
 * consecutive stops: the cells that cleared have refilled from further up the strip while the
 * survivors stayed where they were, so the outline scatters.
 */
function highlightStrips(origins: number[][] | null): void {
  for (let r = 0; r < GAME.reels; r++) {
    for (const cell of stripCells[r]!) cell.classList.remove('in-window')
    if (!origins) continue
    for (const stop of origins[r]!) stripCells[r]![stop]!.classList.add('in-window')
  }
}

// ------------------------------------------------------------------ spinning
const pause = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    if (skip || reducedMotion.matches) {
      resolve()
      return
    }
    window.setTimeout(resolve, ms)
  })

function setBusy(value: boolean): void {
  busy = value
  for (const button of [ui.batch10k, ui.batch1m, ui.reset, ui.seedInput]) {
    button.disabled = value
  }
  ui.spin.textContent = value ? 'Skip' : 'Spin'
}

async function spinOnce(): Promise<void> {
  if (busy) {
    skip = true
    return
  }
  setBusy(true)
  skip = false
  clearWins()

  const stops = sampler.next(new Uint8Array(GAME.reels))
  const trace: CascadeStep[] = []
  const result = evaluator.play(stops, trace)

  let dropping = 0
  for (let i = 0; i < trace.length; i++) {
    const step = trace[i]!
    renderWindow(step.grid, { dropping })
    highlightStrips(step.origins)
    renderLadder(i)
    await pause(SHOW_MS)
    renderWindow(step.grid, { winning: step.cleared })
    appendStepWins(step, i)
    await pause(WIN_MS)
    renderWindow(step.grid, { winning: step.cleared, clearing: step.cleared })
    await pause(CLEAR_MS)
    dropping = refillMask(step.cleared)
  }

  renderWindow(result.finalGrid!, { dropping })
  highlightStrips(result.finalOrigins)
  renderLadder(null)

  if (result.scatterUnits > 0) {
    ui.winsEmpty.hidden = true
    ui.winList.insertAdjacentHTML(
      'beforeend',
      `<li class="win scatter"><span class="win-symbol">${symbolMarkup('S', false)}</span>` +
        `<span class="win-text"><strong>S</strong> ×${result.scatterCount}, ` +
        `anywhere, unmultiplied</span>` +
        `<span class="win-units">${stakes(result.scatterUnits)}</span></li>`,
    )
  }
  ui.spinTotal.textContent =
    result.totalUnits > 0
      ? `This spin paid ${stakes(result.totalUnits)} the stake` +
        (result.steps > 1 ? ` over ${result.steps} cascade steps` : '')
      : 'This spin paid nothing'

  addSpin(committed, result.totalUnits)
  renderFigures()
  setBusy(false)
}

// ------------------------------------------------------------------ batching
function runBatch(spins: number): void {
  if (busy) return
  setBusy(true)
  ui.spin.disabled = true
  pending = emptyTotals()
  ui.progress.textContent = `0 / ${spins.toLocaleString('en-GB')} spins`
  worker.postMessage({
    spins,
    seed,
    stream: STREAM,
    position: sampler.position,
  } satisfies BatchRequest)
}

worker.addEventListener('message', (event: MessageEvent<BatchMessage>) => {
  const message = event.data
  if (message.type === 'progress') {
    pending = message.totals
    renderFigures()
    ui.progress.textContent = `${message.totals.spins.toLocaleString('en-GB')} spins…`
    return
  }
  committed = mergeTotals(committed, message.totals)
  pending = null
  sampler.position = message.position
  ui.progress.textContent = `${message.totals.spins.toLocaleString('en-GB')} spins added`
  renderFigures()
  ui.spin.disabled = false
  setBusy(false)
})

// --------------------------------------------------------------- the session
function restart(): void {
  committed = emptyTotals()
  pending = null
  sampler = new StopSampler(tables, seed, STREAM)
  clearWins()
  ui.progress.textContent = ''
  renderLadder(null)
  renderWindow(openingWindow())
  highlightStrips(strips.map(() => [0, 1, 2]))
  renderFigures()
}

/** Stop 0 on every reel: a window to look at before the first spin, not a spin. */
function openingWindow(): number[][] {
  return strips.map((strip) => [strip[0]!, strip[1]!, strip[2]!])
}

// ------------------------------------------------------------------ wiring
ui.sprite.innerHTML = spriteMarkup(GAME.symbols)
buildStrips()
restart()

ui.spin.addEventListener('click', () => void spinOnce())
ui.batch10k.addEventListener('click', () => runBatch(10_000))
ui.batch1m.addEventListener('click', () => runBatch(1_000_000))
ui.reset.addEventListener('click', restart)
ui.seedInput.addEventListener('change', () => {
  const value = Number(ui.seedInput.value)
  seed = Number.isFinite(value) && value >= 0 ? Math.floor(value) : 2026
  ui.seedInput.value = String(seed)
  restart()
})
window.addEventListener('keydown', (event) => {
  if (event.code === 'Space' && document.activeElement !== ui.seedInput) {
    event.preventDefault()
    void spinOnce()
  }
})
