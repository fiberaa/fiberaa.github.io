export async function createPlannerSession(
  conditionBuffer,
  stepBuffer,
  timeBuffer,
  options,
  gpu,
  solverSteps = [8],
) {
  const worker = new Worker(
    new URL("./planner-worker.js?v=threads-v1", import.meta.url),
    {
      type: "module",
    },
  );
  const pending = new Map();
  let sequence = 0;
  let resolveReady;
  let rejectReady;
  const ready = new Promise((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });
  worker.onmessage = ({ data }) => {
    if (data.type === "ready") {
      resolveReady();
      return;
    }
    const request = pending.get(data.id);
    if (!request) return;
    pending.delete(data.id);
    data.error ? request.reject(new Error(data.error)) : request.resolve(data);
  };
  worker.onerror = (event) => {
    rejectReady(new Error(event.message));
    for (const request of pending.values())
      request.reject(new Error(event.message));
    pending.clear();
  };
  const call = (type, payload, transfer = []) =>
    new Promise((resolve, reject) => {
      const id = ++sequence;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, type, ...payload }, transfer);
    });
  await ready;
  const { inputNames, wasmThreads } = await call(
    "init",
    { conditionBuffer, stepBuffer, timeBuffer, options, gpu, solverSteps },
    [conditionBuffer, stepBuffer, timeBuffer],
  );
  return {
    inputNames,
    wasmThreads,
    sample(payload) {
      return call("sample", payload);
    },
    release() {
      worker.terminate();
    },
  };
}
