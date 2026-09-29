const { chromium, webkit } = require("playwright");
const useWebKit = process.env.BROWSER === "webkit";
const fs = require("fs");
(async () => {
  const b = await (useWebKit ? webkit : chromium).launch(
    useWebKit
      ? { headless: true }
      : {
          channel: "chromium",
          headless: true,
          args: ["--enable-unsafe-webgpu", "--use-angle=metal"],
        },
  );
  fs.mkdirSync("/tmp/fiber-browser-audit", { recursive: true });
  const p = await b.newPage({ viewport: { width: 1400, height: 1000 } });
  p.on("pageerror", (e) => {
    console.error(e);
    process.exit(1);
  });
  await p.goto(
    process.env.FIBER_PREVIEW_URL || "http://localhost:8001/playground/",
  );
  await p.click("#launch");
  await p.waitForFunction(() => window.fiberDebug?.state.ready, null, {
    timeout: 180000,
  });
  const init = await p.evaluate(() => {
    const d = fiberDebug.data,
      c = fiberDebug.sceneConfig,
      a = c.box_qadr,
      dx = d.qpos[0] - d.qpos[a],
      dy = d.qpos[1] - d.qpos[a + 1],
      r = Math.hypot(dx, dy);
    return {
      x: d.qpos[0],
      y: d.qpos[1],
      gx: d.qpos[0] + (1.5 * dx) / r,
      gy: d.qpos[1] + (1.5 * dy) / r,
      yaw: (Math.atan2(dy, dx) * 180) / Math.PI,
    };
  });
  await p.click("[data-task=walk]");
  await p.keyboard.press("w");
  for (const [id, val] of [
    ["x", init.gx],
    ["y", init.gy],
    ["yaw", Math.round(init.yaw)],
  ]) {
    await p.locator("#waypoint-" + id).fill(String(val));
    await p.locator("#waypoint-" + id).dispatchEvent("input");
  }
  await p.waitForFunction(
    () =>
      fiberDebug.state.phase === "goal_reached" ||
      fiberDebug.state.simTime > 25,
    null,
    { timeout: 90000 },
  );
  let s = await p.evaluate(() => ({
    state: fiberDebug.state,
    root: Array.from(fiberDebug.data.qpos.slice(0, 7)),
    events: fiberDebug.events,
  }));
  if (
    s.state.phase !== "goal_reached" ||
    !s.state.locomotionOverride ||
    s.root[2] < 0.4 ||
    s.events.some((e) => e.type === "plan")
  )
    throw Error(JSON.stringify(s));
  const t = s.state.simTime;
  await p.waitForFunction((t) => fiberDebug.state.simTime > t + 1, t);
  for (const degrees of [90, -90]) {
    const yaw = await p.evaluate(() => {
      let q = fiberDebug.data.qpos;
      return (
        (Math.atan2(
          2 * (q[3] * q[6] + q[4] * q[5]),
          1 - 2 * (q[5] ** 2 + q[6] ** 2),
        ) *
          180) /
        Math.PI
      );
    });
    let desired = Math.round(((yaw + degrees + 540) % 360) - 180);
    await p.locator("#waypoint-yaw").fill(String(desired));
    await p.locator("#waypoint-yaw").dispatchEvent("input");
    await p.waitForFunction(
      () => fiberDebug.state.phase === "goal_reached",
      null,
      { timeout: 45000 },
    );
    console.log(
      "TURN PASS",
      degrees,
      await p.evaluate(() => ({
        time: fiberDebug.state.simTime,
        height: fiberDebug.data.qpos[2],
      })),
    );
  }
  await p.locator("#waypoint-height").fill("0.55");
  await p.locator("#waypoint-height").dispatchEvent("input");
  const ht = await p.evaluate(() => fiberDebug.state.simTime);
  await p.waitForFunction((t) => fiberDebug.state.simTime > t + 4, ht, {
    timeout: 20000,
  });
  const h = await p.evaluate(() => ({
    height: fiberDebug.data.qpos[2],
    state: fiberDebug.state,
  }));
  if (
    h.state.locomotionOverride ||
    Math.abs(h.height - 0.55) > 0.08 ||
    h.state.stateDropPlans !== 0
  )
    throw Error("Height fallback failed");
  console.log("HEIGHT PASS", h.height);
  await p.click("#pause");
  await p.screenshot({ path: "/tmp/fiber-browser-audit/bank-walk.png" });
  fs.writeFileSync(
    "/tmp/fiber-browser-audit/bank-walk.json",
    JSON.stringify({ initial: init, ...s, height: h }, null, 2),
  );
  console.log(
    "PASS",
    JSON.stringify({
      time: s.state.simTime,
      root: s.root,
      direct: s.state.locomotionOverride,
      plannerCalls: s.events.filter((e) => e.type === "plan").length,
    }),
  );
  await b.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
