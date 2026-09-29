import { ort } from "./runtime.js";
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./vendor/", self.location.href).href;
let condition, step, clock;
const timeCache = new Map();
async function solverTimes(steps) {
  if (!Number.isInteger(steps) || steps < 2 || steps > 32)
    throw new Error("Planner steps must be an integer between 2 and 32.");
  if (!timeCache.has(steps)) {
    const times = [];
    for (let i = 0; i < steps; i++) {
      const flowTime = new ort.Tensor(
        "float32",
        Float32Array.of(i / steps),
        [1],
      );
      try {
        times.push(await clock.run({ flow_time: flowTime }));
      } finally {
        flowTime.dispose();
      }
    }
    timeCache.set(steps, times);
  }
  return timeCache.get(steps);
}
function tensors(feed) {
  return Object.fromEntries(
    Object.entries(feed).map(([name, v]) => [
      name,
      new ort.Tensor(v.type, v.data, v.dims),
    ]),
  );
}
function dispose(feed) {
  Object.values(feed).forEach((v) => v.dispose());
}
self.onmessage = async ({ data }) => {
  const { id, type } = data;
  try {
    if (type === "init") {
      // Shared WASM memory requires COOP/COEP. Keep the existing one-thread
      // path on hosts without isolation; use the validated four-thread CPU path
      // on the local preview. GPU inference does not need a CPU thread pool.
      ort.env.wasm.numThreads = !data.gpu && self.crossOriginIsolated
        ? Math.min(4, Math.max(1, navigator.hardwareConcurrency || 1)) : 1;
      condition = await ort.InferenceSession.create(data.conditionBuffer, {
        ...data.options,
        ...(data.gpu ? { preferredOutputLocation: "gpu-buffer" } : {}),
      });
      clock = await ort.InferenceSession.create(data.timeBuffer, {
        ...data.options,
        ...(data.gpu ? { preferredOutputLocation: "gpu-buffer" } : {}),
      });
      step = await ort.InferenceSession.create(data.stepBuffer, {
        ...data.options,
        ...(data.gpu ? { preferredOutputLocation: "gpu-buffer" } : {}),
      });
      for (const steps of data.solverSteps || [8]) await solverTimes(steps);
      self.postMessage({
        id,
        inputNames: [...new Set([...condition.inputNames, ...step.inputNames])],
        wasmThreads: ort.env.wasm.numThreads,
      });
      return;
    }
    const input = tensors(data.feed),
      cache = {};
    const started = performance.now();
    try {
      const encoded = await condition.run(
        Object.fromEntries(condition.inputNames.map((n) => [n, input[n]])),
      );
      Object.assign(cache, encoded);
      const conditionMs = performance.now() - started;
      const steps = data.steps;
      const times = await solverTimes(steps);
      let bodyTensor = new ort.Tensor(
        "float32",
        data.trajectory,
        data.trajectoryDims,
      );
      let handTensor = new ort.Tensor(
        "float32",
        data.handTrajectory,
        data.handDims,
      );
      let trajectory, hand;
      try {
        for (let i = 0; i < steps; i++) {
          const remaining = new ort.Tensor(
            "float32",
            Float32Array.of(steps - i),
            [1],
          );
          try {
            const feed = {
              ...input,
              ...cache,
              ...times[i],
              trajectory: bodyTensor,
              hand_trajectory: handTensor,
              remaining_steps: remaining,
            };
            const output = await step.run(
              Object.fromEntries(step.inputNames.map((n) => [n, feed[n]])),
            );
            bodyTensor.dispose();
            handTensor.dispose();
            bodyTensor = output.next_trajectory;
            handTensor = output.next_hand;
          } finally {
            remaining.dispose();
          }
        }
        trajectory = Float32Array.from(await bodyTensor.getData());
        hand = Float32Array.from(await handTensor.getData());
      } finally {
        bodyTensor.dispose();
        handTensor.dispose();
      }
      self.postMessage(
        {
          id,
          trajectory,
          hand,
          conditionMs,
          wasmThreads: ort.env.wasm.numThreads,
          elapsedMs: performance.now() - started,
        },
        [trajectory.buffer, hand.buffer],
      );
    } finally {
      dispose(cache);
      dispose(input);
    }
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
self.postMessage({ type: "ready" });
