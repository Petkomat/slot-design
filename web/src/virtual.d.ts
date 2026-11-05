// The fitted design, supplied at build time by `vite-plugin-design.ts` straight from
// `slotmath/designs.json`.
//
// The type is written with the `import()` operator rather than a nested `import type`: inside a
// `declare module` block TypeScript 7 resolves the latter to `any` without complaining, which
// would quietly strip the checking from every use of the design on the page.
declare module 'virtual:fitted-design' {
  const design: import('./design').FittedDesign
  export default design
}
