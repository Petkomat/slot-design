import { defineConfig } from 'vite'
import { fittedDesignPlugin } from './vite-plugin-design.ts'

// Served from https://petkomat.github.io/slot-design/, so assets resolve under /slot-design/.
export default defineConfig({
  base: process.env.PAGES_BASE ?? '/slot-design/',
  plugins: [fittedDesignPlugin()],
  // The worker is bundled in its own pass, which does not inherit the plugins above; it reads
  // the fitted design too, so it needs the same one.
  worker: { format: 'es', plugins: () => [fittedDesignPlugin()] },
  build: { target: 'es2022', outDir: 'dist', emptyOutDir: true },
  server: { fs: { allow: ['..'] } },
})
