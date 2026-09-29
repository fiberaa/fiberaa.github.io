const { webkit } = require("playwright");
const fs = require("fs");
const path = require("path");
const out =
  process.argv[2] || "/tmp/fiber-browser-audit/webkit-exact-run-grasp";
fs.mkdirSync(out, { recursive: true });
(async () => {
  const b = await webkit.launch({ headless: true });
  const p = await b.newPage();
  p.on("pageerror", (e) => console.log("PAGE ERROR", e.message));
  await p.goto(
    process.env.FIBER_PREVIEW_URL || "http://localhost:8001/playground/",
  );
  await p.click("#launch");
  await p.waitForFunction(() => window.fiberDebug?.state.ready, null, {
    timeout: 180000,
  });
  for (const [i, label] of ["Tangerine", "Water bottle", "Cup"].entries()) {
    await p.click("[data-scene=grasp]");
    await p.waitForFunction(
      () =>
        fiberDebug.state.sceneName === "grasp" &&
        !fiberDebug.state.sceneLoading,
    );
    await p.click(`#grasp-targets [data-object="${i}"]`);
    await p.waitForFunction(
      (label) =>
        !fiberDebug.state.sceneLoading &&
        fiberDebug.sceneConfig.boxes[0].label === label,
      label,
    );
    await p.click("#reset");
    await p.click("#pause");
    await p.waitForFunction(() => fiberDebug.state.simTime >= 3, null, {
      timeout: 180000,
    });
    console.log(
      "SWITCH",
      label,
      await p.evaluate(() => ({
        time: fiberDebug.state.simTime,
        last: fiberDebug.state.lastExecutedCommand,
      })),
    );
    await p.click("[data-task=grasp]");
    await p.waitForFunction(() => !fiberDebug.state.running, null, {
      timeout: 360000,
    });
    await p.locator("details.help summary").click();
    const [download] = await Promise.all([
      p.waitForEvent("download"),
      p.click("#save-run-report"),
    ]);
    const name = label.toLowerCase().replaceAll(" ", "-");
    await download.saveAs(path.join(out, name + ".json"));
    await p.screenshot({ path: path.join(out, name + ".png") });
    const r = JSON.parse(fs.readFileSync(path.join(out, name + ".json")));
    const s = r.state;
    console.log("RESULT", label, {
      time: s.simTime,
      lift: s.maxLift,
      hold: s.heldFrames,
      contact: s.contactFrames,
      drop: s.stateDropPlans,
      events: r.events.length,
    });
    await p.locator("details.help summary").click();
  }
  await b.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
