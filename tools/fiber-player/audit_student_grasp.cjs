const { chromium, webkit } = require("playwright");
const fs = require("fs");
const out = process.argv[2] || "/tmp/fiber-browser-audit/current-grasp";
fs.mkdirSync(out, { recursive: true });
(async () => {
  let failures = 0;
  const useWK = process.env.BROWSER === "webkit";
  const b = await (useWK ? webkit : chromium).launch(
    useWK
      ? { headless: true }
      : {
          channel: "chromium",
          headless: true,
          args: ["--enable-unsafe-webgpu", "--use-angle=metal"],
        },
  );
  const p = await b.newPage({ viewport: { width: 1400, height: 900 } });
  try {
    p.on("pageerror", (e) => console.log("PAGEERROR", e.message));
    await p.goto(
      process.env.FIBER_PREVIEW_URL || "http://localhost:8001/playground/",
    );
    await p.click("#launch");
    await p.waitForFunction(() => window.fiberDebug?.state.ready, null, {
      timeout: 180000,
    });
    for (let i = 0; i < 3; i++) {
      if (await p.evaluate(() => fiberDebug.state.running))
        await p.click("#pause");
      await p.click("[data-scene=grasp]");
      await p.waitForFunction(
        () =>
          fiberDebug.state.sceneName === "grasp" &&
          !fiberDebug.state.sceneLoading,
      );
      await p.click(`#grasp-targets [data-object="${i}"]`);
      await p.waitForFunction(
        (i) =>
          !fiberDebug.state.sceneLoading &&
          fiberDebug.sceneConfig.boxes[0].label ===
            ["Tangerine", "Water bottle", "Cup"][i],
        i,
      );
      if (process.env.WARM) {
        await p.click("#pause");
        await p.waitForFunction(() => fiberDebug.state.simTime >= 3, null, {
          timeout: 30000,
        });
      }
      await p.click("[data-task=grasp]");
      const start = await p.evaluate(() => fiberDebug.state.simTime);
      await p.waitForFunction(
        (t) =>
          fiberDebug.state.phase === "holding_object" ||
          fiberDebug.state.simTime >= t + 20 ||
          !fiberDebug.state.running,
        start,
        { timeout: 180000 },
      );
      await p.click("#pause");
      const r = await p.evaluate(() => ({
        state: fiberDebug.state,
        root: Array.from(fiberDebug.data.qpos.slice(0, 7)),
        rootStart: fiberDebug.sceneConfig.qpos.slice(0, 7),
        provenance: fiberDebug.sceneConfig.provenance,
        hands: fiberDebug.sceneConfig.hand.qadr.map(
          (a) => fiberDebug.data.qpos[a],
        ),
        events: fiberDebug.events,
        label: fiberDebug.sceneConfig.boxes[0].label,
        objectCenter: Array.from(
          fiberDebug.data.geom_xpos.slice(
            fiberDebug.sceneConfig.object_geom * 3,
            fiberDebug.sceneConfig.object_geom * 3 + 3,
          ),
        ),
      }));
      r.pass =
        r.state.phase === "holding_object" &&
        r.state.solverSteps === 3 &&
        r.state.planningHz === 1 &&
        !r.state.rtc.enabled &&
        r.state.stateDropPlans === (process.env.WARM ? 1 : 0);
      if (!r.pass) failures++;
      fs.writeFileSync(`${out}/${i}.json`, JSON.stringify(r, null, 2));
      await p.screenshot({ path: `${out}/${i}.png` });
      console.log(
        "RESULT",
        r.label,
        r.pass,
        r.state.simTime,
        r.state.maxLift,
        r.state.contactFrames,
      );
    }
  } finally {
    await b.close();
  }
  if (failures) process.exitCode = 1;
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
