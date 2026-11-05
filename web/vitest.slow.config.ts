import { defineConfig } from 'vitest/config'
import { fittedDesignPlugin } from './vite-plugin-design.ts'

// The slow tier: one million spins through the TypeScript evaluator, compared with the totals
// `slotmath/webfixture.py` recorded for the same seeded stream.
export default defineConfig({
  plugins: [fittedDesignPlugin()],
  test: { include: ['test/**/*.slow.test.ts'], testTimeout: 600_000 },
})
