const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");
const output = process.argv[2] || "/tmp/fiber-browser-audit/defaults";
fs.mkdirSync(output, { recursive: true });
const cases = [
  ...["ground", "support", "throw"].map((layout) => ({
    id: `pickup-${layout}`,
    scene: "pickup",
    layout,
  })),
  { id: "door", scene: "door" },
  ...["Tangerine", "Water bottle", "Cup"].map((label, object) => ({
    id: `grasp-${object + 1}`,
    scene: "grasp",
    object,
    label,
  })),
];
if (process.env.CASES) {
  const chosen = process.env.CASES.split(",");
  for (let i = cases.length - 1; i >= 0; i--)
    if (!chosen.includes(cases[i].id)) cases.splice(i, 1);
}
if (process.env.PICKUP_ONLY) cases.splice(3);
const log = path.join(output, "results.jsonl");
fs.writeFileSync(log, "");
let next = 0;
(async () => {
  const browser = await chromium.launch({
    channel: "chromium",
    headless: true,
    args: ["--enable-unsafe-webgpu", "--use-angle=metal"],
  });
  async function worker(index) {
    const page = await browser.newPage({
      viewport: { width: 1400, height: 900 },
    });
    let errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(
      process.env.FIBER_PREVIEW_URL || "http://localhost:8001/playground/",
    );
    await page.click("#launch");
    await page.waitForFunction(() => window.fiberDebug?.state.ready, null, {
      timeout: 180000,
    });
    while (next < cases.length) {
      const c = cases[next++];
      errors = [];
      const wall = Date.now();
      console.log("START", index, c.id);
      try {
        if (await page.evaluate(() => fiberDebug.state.running))
          await page.click("#pause");
        await page.click(`[data-scene=${c.scene}]`);
        await page.waitForFunction(
          (name) =>
            fiberDebug.state.sceneName === name &&
            !fiberDebug.state.sceneLoading,
          c.scene,
          { timeout: 30000 },
        );
        if (c.layout) {
          await page.click(`[data-layout=${c.layout}]`);
          await page.waitForFunction(
            (layout) =>
              !fiberDebug.state.sceneLoading &&
              fiberDebug.state.pickupLayout === layout,
            c.layout,
          );
        }
        if (c.scene === "grasp") {
          await page.click(`#grasp-targets [data-object="${c.object}"]`);
          await page.waitForFunction(
            (label) =>
              !fiberDebug.state.sceneLoading &&
              fiberDebug.sceneConfig.boxes[0].label === label,
            c.label,
          );
        }
        const initial = await page.evaluate(() => ({
          state: fiberDebug.state,
          root: Array.from(fiberDebug.data.qpos.slice(0, 7)),
          dimensions: fiberDebug.sceneConfig.boxes?.[0]?.dimensions,
          geometry: Array.from(
            fiberDebug.model.geom_size.slice(
              fiberDebug.sceneConfig.object_geom * 3,
              fiberDebug.sceneConfig.object_geom * 3 + 3,
            ),
          ),
        }));
        if (c.scene === "pickup") {
          const spawn = initial.state.pickupSpawn;
          if (
            !spawn ||
            spawn.radius < 0.9 ||
            spawn.radius > 1.5 ||
            initial.state.stateDropPending
          )
            throw new Error("Invalid pickup reset or startup state dropout");
          if (
            initial.dimensions.some(
              (v, i) => Math.abs(v / 2 - initial.geometry[i]) > 1e-6,
            )
          )
            throw new Error("Box collision dimensions do not match selection");
        }
        if (c.layout === "throw") {
          const point = await page.evaluate(() => fiberDebug.objectScreenPoint);
          await page.mouse.move(point.x, point.y);
          await page.mouse.down();
          await page.mouse.move(point.x + 65, point.y - 45, { steps: 10 });
          await page.mouse.up();
          await page.waitForFunction(() =>
            fiberDebug.events.some((e) => e.type === "throw"),
          );
        } else await page.click(`[data-task=${c.scene}]`);
        await page.waitForFunction(
          () => {
            const s = fiberDebug.state;
            return (
              !s.running ||
              s.simTime >= 45 ||
              fiberDebug.data.qpos[2] < 0.25 ||
              (s.sceneName === "pickup" && s.phase === "holding") ||
              (s.sceneName === "door" && s.phase === "complete") ||
              (s.sceneName === "grasp" &&
                s.heldFrames >= 51 &&
                s.contactFrames >= 45)
            );
          },
          null,
          { timeout: 360000 },
        );
        const result = await page.evaluate(() => {
          const s = fiberDebug.state,
            cfg = fiberDebug.sceneConfig,
            d = fiberDebug.data;
          return {
            state: s,
            events: fiberDebug.events,
            root: Array.from(d.qpos.slice(0, 7)),
            objectCenter: Array.from(
              d.geom_xpos.slice(cfg.object_geom * 3, cfg.object_geom * 3 + 3),
            ),
            hinge:
              cfg.hinge_qadr === null || cfg.hinge_qadr === undefined
                ? null
                : d.qpos[cfg.hinge_qadr],
            provenance: cfg.provenance,
            sampler: cfg.sampler_noise || s.sceneName,
          };
        });
        const pass =
          c.scene === "pickup"
            ? result.state.holdingBox && result.state.phase === "holding"
            : c.scene === "door"
              ? result.state.phase === "complete" &&
                result.hinge > 0.5 &&
                result.root[0] > 1.6
              : result.state.heldFrames >= 51 &&
                result.state.contactFrames >= 45 &&
                result.state.maxLift > 0.1;
        const row = {
          ...c,
          initial,
          pass,
          wallSeconds: (Date.now() - wall) / 1000,
          errors,
          ...result,
        };
        fs.appendFileSync(log, JSON.stringify(row) + "\n");
        console.log(
          "RESULT",
          c.id,
          pass,
          result.state.simTime,
          result.state.phase,
          result.state.error,
        );
        await page.screenshot({ path: path.join(output, c.id + ".png") });
        if (result.state.running) await page.click("#pause");
      } catch (e) {
        const row = {
          ...c,
          pass: false,
          error: String(e),
          errors,
          wallSeconds: (Date.now() - wall) / 1000,
        };
        fs.appendFileSync(log, JSON.stringify(row) + "\n");
        console.log("ERROR", c.id, e.message);
      }
    }
    await page.close();
  }
  await Promise.all([worker(0), worker(1)]);
  await browser.close();
  const rows = fs.readFileSync(log, "utf8").trim().split("\n").map(JSON.parse);
  console.log("TOTAL", rows.filter((r) => r.pass).length + "/" + rows.length);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
