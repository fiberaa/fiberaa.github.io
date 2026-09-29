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
    const p = await b.newPage();
    const errors = [];
    p.on("pageerror", (e) => errors.push(e.message));
    await p.addInitScript(() => {
      let seed = 314159;
      Math.random = () =>
        (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
    });
    await p.goto(process.env.FIBER_URL || "http://localhost:8001/playground/");
    await p.click("#launch");
    await p.waitForFunction(() => window.fiberDebug?.state.ready, null, {
      timeout: 180000,
    });
    await p.evaluate(() => {
      window.trace = [];
      window.traceTimer = setInterval(() => {
        const s = fiberDebug.state;
        trace.push({
          wall: performance.now(),
          sim: s.simTime,
          pending: s.rtc.pending,
          buffer: s.rtc.bufferedControls,
          accepted: s.rtc.acceptedPlans,
          held: s.rtc.heldControls,
        });
      }, 20);
    });
    await p.click("[data-task=pickup]");
    await p.waitForFunction(() => fiberDebug.state.simTime >= 6, null, {
      timeout: 180000,
    });
    await p.click("#pause");
    await p.waitForTimeout(100);
    const first = await p.evaluate(() => {
      clearInterval(traceTimer);
      return { state: fiberDebug.state, events: fiberDebug.events, trace };
    });
    const plans = first.events.filter((e) => e.type === "plan");
    if (!plans.length || plans[0].droppedState)
      throw Error("Missing initial plan or initial state dropout");
    if (
      first.state.solverSteps !== 3 ||
      first.state.plannerModel !== "v27-distill2000"
    )
      throw Error("Wrong planner");
    let pendingProgress = 0;
    for (let i = 1; i < first.trace.length; i++)
      if (first.trace[i].pending && first.trace[i - 1].pending)
        pendingProgress += Math.max(
          0,
          first.trace[i].sim - first.trace[i - 1].sim,
        );
    if (pendingProgress < 0.06)
      throw Error("Simulation did not advance during planning");
    console.log("RTC", wk ? "webkit" : "chromium", {
      pendingProgress,
      phase: first.state.phase,
      rtc: first.state.rtc,
    });
    // Reset while inference is in flight must invalidate its result.
    await p.click("#pause");
    await p.waitForFunction(() => fiberDebug.state.rtc.pending, null, {
      timeout: 30000,
    });
    await p.click("#reset");
    await p.waitForFunction(
      () =>
        fiberDebug.state.lastExecutedCommand === "walk" &&
        fiberDebug.state.simTime >= 0.4,
      null,
      { timeout: 30000 },
    );
    await p.click("[data-task=pickup]");
    await p.waitForFunction(
      () =>
        fiberDebug.events.filter(
          (e) => e.type === "plan" && e.command === "pickup",
        ).length >= 2,
      null,
      { timeout: 90000 },
    );
    await p.click("#pause");
    await p.waitForTimeout(100);
    const switched = await p.evaluate(() => ({
      state: fiberDebug.state,
      events: fiberDebug.events,
    }));
    const accepted = switched.events.filter(
      (e) => e.type === "plan" && e.command === "pickup",
    );
    if (
      !accepted[0].droppedState ||
      accepted.slice(1).some((e) => e.droppedState) ||
      switched.state.stateDropPlans !== 1
    )
      throw Error("Switch dropout did not occur exactly once");
    if (errors.length) throw Error(errors.join("\n"));
    fs.mkdirSync("/tmp/fiber-browser-audit/rtc", { recursive: true });
    fs.writeFileSync(
      `/tmp/fiber-browser-audit/rtc/${wk ? "webkit" : "chromium"}.json`,
      JSON.stringify({ first, switched, pendingProgress, errors }, null, 2),
    );
    console.log("PASS RTC reset/switch", wk ? "webkit" : "chromium");
  } finally {
    await b.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
