// How the ten symbols are drawn. Geometry and colour, no bitmaps: every symbol has both a
// distinct silhouette and a distinct hue, and carries its name, so a window can be read off the
// screen and checked against the strips without guessing which fruit is which.

export interface SymbolStyle {
  /** Path in a 0..100 viewBox, or a primitive element. */
  shape: string
  colour: string
}

const PATHS: Record<string, string> = {
  // Three highs: many-pointed, hexagon, pentagon.
  H1: '<path d="M50 4 L61 33 L92 22 L80 50 L92 78 L61 67 L50 96 L39 67 L8 78 L20 50 L8 22 L39 33 Z"/>',
  H2: '<path d="M50 6 L88 28 L88 72 L50 94 L12 72 L12 28 Z"/>',
  H3: '<path d="M50 8 L92 38 L76 88 L24 88 L8 38 Z"/>',
  // Five lows: circle, square, triangle, diamond, chevron.
  L1: '<circle cx="50" cy="50" r="38"/>',
  L2: '<rect x="14" y="14" width="72" height="72" rx="14"/>',
  L3: '<path d="M50 12 L90 84 L10 84 Z"/>',
  L4: '<path d="M50 8 L90 50 L50 92 L10 50 Z"/>',
  L5: '<path d="M12 26 L50 52 L88 26 L88 50 L50 76 L12 50 Z"/>',
  // The wild: an open ring, the only hollow symbol.
  W: '<circle cx="50" cy="50" r="34" fill="none" stroke="currentColor" stroke-width="14"/>',
  // The scatter: the only five-pointed star, and the only symbol that pays from anywhere.
  S: '<path d="M50 6 L61 38 L95 38 L68 58 L78 90 L50 70 L22 90 L32 58 L5 38 L39 38 Z"/>',
}

const COLOURS: Record<string, string> = {
  H1: '#f2b134',
  H2: '#e8653f',
  H3: '#d4529a',
  L1: '#4aa3df',
  L2: '#3fb89a',
  L3: '#7f7bd6',
  L4: '#92a6b2',
  L5: '#7f9b43',
  W: '#ffd264',
  S: '#b084f5',
}

export const symbolColour = (name: string): string => COLOURS[name] ?? '#888'

/** The sprite every cell points at, injected once. */
export function spriteMarkup(symbols: readonly string[]): string {
  const defs = symbols
    .map(
      (name) =>
        `<symbol id="sym-${name}" viewBox="0 0 100 100">${PATHS[name] ?? PATHS.L1}</symbol>`,
    )
    .join('')
  return `<svg class="sprite" aria-hidden="true"><defs>${defs}</defs></svg>`
}

/** One symbol, as it appears in a reel cell or a strip column. */
export function symbolMarkup(name: string, withLabel = true): string {
  const colour = symbolColour(name)
  const label = withLabel ? `<span class="cell-label">${name}</span>` : ''
  return (
    `<svg class="cell-art" viewBox="0 0 100 100" role="img" aria-label="${name}" ` +
    `style="color:${colour};fill:${colour}"><use href="#sym-${name}"/></svg>${label}`
  )
}
