const { chromium } = require("playwright");
const fs = require("fs");
const backend = process.env.BACKEND || "webgpu";
(async () => {
  const rows = [];
  for (const teacher of [true, false]) {
    const browser = await chromium.launch({
      channel: "chromium",
      headless: true,
      args: ["--enable-unsafe-webgpu", "--use-angle=metal"],
    });
    const page = await browser.newPage();
    page.on("pageerror", (e) => console.log("PAGE ERROR", e.message));
    await page.goto(
      `http://localhost:8001/playground/?backend=${backend}${teacher ? "&planner=teacher" : ""}`,
    );
    await page.click("#launch");
    await page.waitForFunction(() => window.fiberDebug?.state.ready, null, {
      timeout: 240000,
    });
    await page.click("[data-scene=grasp]");
    await page.waitForFunction(
      () =>
        fiberDebug.state.sceneName === "grasp" &&
        !fiberDebug.state.sceneLoading,
    );
    await page.evaluate(() => {
      const original = fiberDebug.planner.sample.bind(fiberDebug.planner);
      fiberDebug.planner.sample = async (payload) => {
        window.benchmarkPayload ??= payload;
        return original(payload);
      };
    });
    await page.click("[data-task=grasp]");
    await page.waitForFunction(() => fiberDebug.state.simTime >= 0.02, null, {
      timeout: 180000,
    });
    await page.click("#pause");
    await page.waitForTimeout(1000);
    const result = await page.evaluate(async () => {
      const timing = [];
      let output;
      for (let i = 0; i < 7; i++) {
        const start = performance.now();
        output = await fiberDebug.planner.sample(benchmarkPayload);
        timing.push({
          wallMs: performance.now() - start,
          workerMs: output.elapsedMs,
          conditionMs: output.conditionMs,
        });
      }
      return {
        state: fiberDebug.state,
        steps: benchmarkPayload.steps,
        timing,
        finite:
          Array.from(output.trajectory).every(Number.isFinite) &&
          Array.from(output.hand).every(Number.isFinite),
      };
    });
    rows.push(result);
    console.log(
      "BENCH",
      teacher ? "teacher" : "student",
      backend,
      result.steps,
      result.timing.slice(2),
      result.finite,
    );
    await browser.close();
  }
  fs.writeFileSync(
    `/tmp/fiber-browser-audit/distill-benchmark-${backend}.json`,
    JSON.stringify(rows, null, 2),
  );
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
