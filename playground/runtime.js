// Use native stack switching in the planner worker when the browser supports it.
export const ort = await import(
  typeof document === "undefined" &&
  typeof WebAssembly.Suspending === "function"
    ? "./vendor/ort.jspi.bundle.min.mjs"
    : "./vendor/ort.webgpu.bundle.min.mjs"
);
