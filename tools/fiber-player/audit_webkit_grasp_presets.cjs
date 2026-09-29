const { webkit } = require("playwright"),
  fs = require("fs");
(async () => {
  const b = await webkit.launch({ headless: true });
  try {
    const p = await b.newPage();
    p.on("pageerror", (e) => console.log("PAGEERROR", e.message));
    await p.goto("http://localhost:8001/playground/");
    await p.click("#launch");
    await p.waitForFunction(() => window.fiberDebug?.state.ready, null, {
      timeout: 180000,
    });
    await p.click("[data-scene=grasp]");
    await p.waitForFunction(
      () =>
        fiberDebug.state.sceneName === "grasp" &&
        !fiberDebug.state.sceneLoading,
    );
    const trials = [{ i: 2 }, { i: 1 }, { i: 0 }];
    const results = [];
    let passed = false;
    for (const c of trials) {
      await p.click(`#grasp-targets [data-object="${c.i}"]`);
      await p.waitForFunction(
        (i) =>
          !fiberDebug.state.sceneLoading &&
          fiberDebug.sceneConfig.boxes[0].label ===
            ["Tangerine", "Water bottle", "Cup"][i],
        c.i,
      );
      await p.click("[data-task=grasp]");
      await p.waitForFunction(
        () =>
          fiberDebug.state.phase === "holding_object" ||
          fiberDebug.state.simTime >= 20,
        null,
        { timeout: 240000 },
      );
      await p.click("#pause");
      const r = await p.evaluate(() => fiberDebug.state);
      results.push({ trial: c, ...r });
      passed = r.phase === "holding_object";
      console.log("GRASP", c, r.phase, r.simTime, r.maxLift, r.contactFrames);
      fs.mkdirSync("/tmp/fiber-browser-audit/webkit-grasp-presets", {
        recursive: true,
      });
      fs.writeFileSync(
        "/tmp/fiber-browser-audit/webkit-grasp-presets/results.json",
        JSON.stringify(results, null, 2),
      );
    }
  } finally {
    await b.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
