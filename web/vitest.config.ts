import { defineConfig } from 'vitest/config'
import { fittedDesignPlugin } from './vite-plugin-design.ts'

// The fast tier, run on every push: the 20,000-spin agreement check and the unit tests.
// The million-spin check lives in vitest.slow.config.ts, mirroring pytest's "slow" marker.
export default defineConfig({
  plugins: [fittedDesignPlugin()],
  test: { include: ['test/**/*.test.ts'], exclude: ['test/**/*.slow.test.ts'] },
})
