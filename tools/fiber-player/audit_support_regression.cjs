const { chromium, webkit } = require("playwright"),
  fs = require("fs");
(async () => {
  const wk = process.env.BROWSER === "webkit";
  const b = await (wk ? webkit : chromium).launch(
    wk
      ? { headless: true }
      : {
          channel: "chromium",
          headless: true,
          args: ["--enable-unsafe-webgpu", "--use-angle=metal"],
        },
  );
  try {
    const p = await b.newPage({ viewport: { width: 1400, height: 950 } });
    p.on("pageerror", (e) => console.log("PAGEERROR", e.message));
    await p.addInitScript(() => {
      let seed = 314159;
      Math.random = () =>
        (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
    });
    await p.goto(
      process.env.FIBER_URL ||
        "http://localhost:8001/playground/?planner=teacher&backend=wasm",
    );
    await p.click("#launch");
    await p.waitForFunction(() => window.fiberDebug?.state.ready, null, {
      timeout: 180000,
    });
    const reports = [];
    for (const layout of ["support"]) {
      for (let i = 0; i < 1; i++) {
        if (await p.evaluate(() => fiberDebug.state.running))
          await p.click("#pause");
        await p.click(`[data-layout=${layout}]`);
        await p.waitForFunction(
          (layout) =>
            fiberDebug.state.pickupLayout === layout &&
            !fiberDebug.state.sceneLoading,
          layout,
        );
        await p.evaluate(() => {
          const original = Math.random;
          const values = [
            (1.208286970295012 - 0.9) / 0.6,
            (0.041109529174360206 / ((65 * Math.PI) / 180) + 1) / 2,
            (-0.06945045358603244 / ((20 * Math.PI) / 180) + 1) / 2,
          ];
          Math.random = () => (values.length ? values.shift() : original());
          fiberDebug.reset(false);
          Math.random = original;
        });
        const initial = await p.evaluate(() => fiberDebug.state);
        await p.click("[data-task=pickup]");
        await p.waitForFunction(
          () =>
            fiberDebug.state.holdingBox ||
            fiberDebug.state.simTime >= 20 ||
            fiberDebug.data.qpos[2] < 0.25,
          null,
          { timeout: 240000 },
        );
        await p.click("#pause");
        const r = await p.evaluate(() => ({
          state: fiberDebug.state,
          root: Array.from(fiberDebug.data.qpos.slice(0, 7)),
          events: fiberDebug.events,
        }));
        reports.push({ layout, initial, ...r });
        console.log(
          "BOX",
          layout,
          i,
          r.state.holdingBox,
          "time",
          r.state.simTime,
          "lift",
          r.state.maxLift,
          "contact",
          r.state.contactFrames,
          "spawn",
          initial.pickupSpawn,
        );
      }
    }
    fs.mkdirSync("/tmp/fiber-browser-audit/support-regression-cpu", {
      recursive: true,
    });
    fs.writeFileSync(
      `/tmp/fiber-browser-audit/support-regression-cpu/${wk ? "webkit" : "chromium"}.json`,
      JSON.stringify(reports, null, 2),
    );
  } finally {
    await b.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
