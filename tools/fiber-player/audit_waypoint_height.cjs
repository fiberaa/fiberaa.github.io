const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");
const output = process.argv[2] || "/tmp/fiber-browser-audit/waypoint-height";
fs.mkdirSync(output, { recursive: true });
(async () => {
  const b = await chromium.launch({
    channel: "chromium",
    headless: true,
    args: ["--enable-unsafe-webgpu", "--use-angle=metal"],
  });
  const p = await b.newPage({ viewport: { width: 1400, height: 1000 } });
  await p.goto(
    process.env.FIBER_PREVIEW_URL || "http://localhost:8001/playground/",
  );
  await p.click("#launch");
  await p.waitForFunction(() => window.fiberDebug?.state.ready, null, {
    timeout: 180000,
  });
  await p.click("[data-task=walk]");
  await p.keyboard.press("w");
  await p.locator("#waypoint-height").fill("0.55");
  await p.locator("#waypoint-height").dispatchEvent("input");
  await p.waitForFunction(
    () => fiberDebug.state.simTime >= 4 || !fiberDebug.state.running,
    null,
    { timeout: 90000 },
  );
  if (await p.evaluate(() => fiberDebug.state.running)) await p.click("#pause");
  const r = await p.evaluate(() => ({
    state: fiberDebug.state,
    command: Array.from(fiberDebug.command()),
    root: Array.from(fiberDebug.data.qpos.slice(0, 7)),
    events: fiberDebug.events,
  }));
  console.log(JSON.stringify(r));
  if (
    Math.abs(r.command[4] - 0.55) > 0.02 ||
    Math.abs(r.command[9] - 0.55) > 0.02 ||
    r.state.stateDropPlans !== 0 ||
    Math.abs(r.root[2] - 0.55) > 0.08
  )
    throw Error("Height conditioning, physical height, or dropout failure");
  await p.screenshot({ path: path.join(output, "height-control.png") });
  fs.writeFileSync(
    path.join(output, "height-control.json"),
    JSON.stringify(r, null, 2),
  );
  await b.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
