import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import {
  locomotionZ,
  validateBank,
  blend,
  smoothLocomotionZ,
} from "./locomotion.js?v=20260929-door-approach";
import loadMujoco from "./vendor/mujoco.js";
import { ort } from "./runtime.js";
import { createPlannerSession } from "./planner-session.js?v=threads-v1";

const useF16Storage =
  new URLSearchParams(location.search).get("precision") === "f16";
const useTeacher =
  new URLSearchParams(location.search).get("planner") === "teacher" ||
  useF16Storage;
const plannerModel = useF16Storage
  ? "v27-f16-storage"
  : useTeacher
    ? "v27"
    : "v27-distill2000";
const weightSuffix = useF16Storage ? "f16" : "w8";
const auditSteps = Number(new URLSearchParams(location.search).get("steps"));
let stepOverride =
  Number.isInteger(auditSteps) && auditSteps >= 2 && auditSteps <= 32
    ? auditSteps
    : null;
const solverSteps = stepOverride
  ? [stepOverride]
  : useTeacher
    ? [4, 8, 16]
    : [3];
const controlsPerPlan = 50; // 50 Hz actor × one second between planner calls.
const $ = (s) => document.querySelector(s),
  sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const canvas = $("#scene"),
  viewport = $("#viewport");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.shadowMap.enabled = true;
const world = new THREE.Scene();
world.background = new THREE.Color("#182436");
world.fog = new THREE.Fog("#182436", 15, 40);
const camera = new THREE.PerspectiveCamera(43, 1, 0.05, 80);
camera.up.set(0, 0, 1);
camera.position.set(-3.5, -5, 3.6);
const orbit = new OrbitControls(camera, canvas);
orbit.target.set(1, 0, 0.65);
orbit.enableDamping = true;
orbit.maxDistance = 20;
orbit.minDistance = 1.3;
orbit.maxPolarAngle = Math.PI * 0.49;
world.add(new THREE.HemisphereLight(0xe8f1ff, 0x41556c, 1.1));
const sunlight = new THREE.DirectionalLight(0xffffff, 1.3);
sunlight.position.set(-3, -4, 9);
sunlight.castShadow = true;
sunlight.shadow.mapSize.set(2048, 2048);
Object.assign(sunlight.shadow.camera, {
  left: -9,
  right: 9,
  top: 9,
  bottom: -9,
  near: 0.1,
  far: 25,
});
sunlight.shadow.bias = -0.00015;
world.add(sunlight);
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(100, 100),
  new THREE.MeshStandardMaterial({ color: 0x354558, roughness: 0.95 }),
);
floor.position.z = 0;
floor.receiveShadow = true;
world.add(floor);
const grid = new THREE.GridHelper(30, 60, 0x435166, 0x354457);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.001;
world.add(grid);
const marker = new THREE.Mesh(
  new THREE.RingGeometry(0.18, 0.23, 40),
  new THREE.MeshBasicMaterial({ color: 0xb6acff, side: THREE.DoubleSide }),
);
marker.position.z = 0.008;
marker.visible = false;
world.add(marker);
const heightStem = new THREE.Line(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({
    color: 0xb6acff,
    transparent: true,
    opacity: 0.55,
  }),
);
heightStem.visible = false;
world.add(heightStem);
// These overlays show the actual command token, never the policy's depth input.
const waypointGuide = new THREE.Group();
waypointGuide.visible = false;
world.add(waypointGuide);
const waypointPath = new THREE.Line(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0x8eabd2 }),
);
waypointGuide.add(waypointPath);
const waypointViews = [0xa8c6e9, 0xb6acff].map((color, i) => {
  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(0.045, 12, 8),
    new THREE.MeshBasicMaterial({ color }),
  );
  const arrow = new THREE.ArrowHelper(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(),
    0.38 + i * 0.12,
    color,
    0.1,
    0.06,
  );
  for (const part of [arrow.line, arrow.cone]) {
    part.material.depthTest = false;
    part.material.depthWrite = false;
    part.renderOrder = 20 + i;
  }
  const labelCanvas = document.createElement("canvas");
  labelCanvas.width = 160;
  labelCanvas.height = 64;
  const context = labelCanvas.getContext("2d");
  context.fillStyle = "rgba(15, 23, 35, .85)";
  context.beginPath();
  context.roundRect(0, 0, 160, 64, 18);
  context.fill();
  context.fillStyle = "#fff";
  context.font = "28px sans-serif";
  context.textAlign = "center";
  context.fillText(`${[0.5, 1.0][i].toFixed(1)} s`, 80, 43);
  const label = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: new THREE.CanvasTexture(labelCanvas),
      depthTest: false,
    }),
  );
  label.scale.set(0.23, 0.092, 1);
  waypointGuide.add(dot, arrow, label);
  return { dot, arrow, label };
});
const headingArrow = new THREE.ArrowHelper(
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(),
  0.45,
  0xffffff,
  0.12,
  0.065,
);
for (const part of [headingArrow.line, headingArrow.cone]) {
  part.material.depthTest = false;
  part.material.depthWrite = false;
  part.renderOrder = 19;
}
headingArrow.visible = false;
world.add(headingArrow);
let lastExecutedCommand = null,
  stateDropPlans = 0;
function shouldDropState(command = mode) {
  return (
    simTime > 0 &&
    lastExecutedCommand !== null &&
    command !== lastExecutedCommand &&
    !holdingBox
  );
}
let pickupSettleUntil = 0,
  graspSettleUntil = 0,
  holdingLossFrames = 0;
let graspSwitchSamplerActive = false;
let carryRouteStage = 0;
let doorHandContactStats = {
  leftFrames: 0,
  rightFrames: 0,
  leftImpulseNs: 0,
  rightImpulseNs: 0,
};
let sceneLoading = false,
  navigationIntent = "walk",
  holdingBox = false;
let conditionedWaypoints = [];
let walkSelected = false,
  yawRate = 1.8,
  goalRevision = 0,
  pickupLayout = "ground",
  activeBox = 0;
let goalHeight = 0.75,
  filteredHeight = 0.75;
let filteredGoal = null,
  filteredYaw = null,
  lastAnimation = performance.now();
const heldKeys = new Map();
const targetArrow = new THREE.ArrowHelper(
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(),
  0.55,
  0xb6acff,
  0.13,
  0.075,
);
targetArrow.visible = false;
marker.add(targetArrow);
function walkMode(enabled) {
  walkSelected = enabled;
  if (enabled)
    document
      .querySelectorAll("[data-task]")
      .forEach((b) => b.classList.remove("active"));
  $(".waypoint-editor").hidden = !enabled;
  const throwing = sceneName === "pickup" && pickupLayout === "throw";
  orbit.mouseButtons.LEFT = enabled || throwing ? null : THREE.MOUSE.ROTATE;
  orbit.mouseButtons.RIGHT = THREE.MOUSE.ROTATE;
  canvas.style.cursor = enabled ? "crosshair" : "grab";
  $("#scene-hint").textContent = enabled
    ? "Click to place · drag to face · right-drag to orbit"
    : throwing
      ? "Drag the box to throw · right-drag to orbit"
      : "Drag to orbit · scroll to zoom";
  $("[data-task='walk']").classList.toggle(
    "active",
    enabled && navigationIntent === "walk",
  );
  $("[data-task='carry']").classList.toggle(
    "active",
    enabled && navigationIntent === "carry",
  );
}
function syncGoal() {
  for (const [id, value] of [
    ["waypoint-x", goal?.[0]],
    ["waypoint-y", goal?.[1]],
    ["waypoint-height", goalHeight],
    ["waypoint-yaw", goalYaw === null ? null : (goalYaw * 180) / Math.PI],
  ]) {
    const input = $("#" + id);
    input.disabled = !goal;
    if (
      value !== undefined &&
      value !== null &&
      document.activeElement !== input
    )
      input.value = value.toFixed(id === "waypoint-yaw" ? 0 : 2);
  }
  $("#waypoint-height-value").textContent = `${goalHeight.toFixed(2)} m`;
  $("#waypoint-yaw-value").textContent =
    `${Math.round(((goalYaw ?? 0) * 180) / Math.PI)}°`;
  $("#waypoint-mode").textContent = goal
    ? "Updates while moving"
    : "Click the floor to place a goal";
  if (goal) {
    marker.position.set(goal[0], goal[1], goalHeight);
    heightStem.geometry.dispose();
    heightStem.geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(goal[0], goal[1], 0.015),
      new THREE.Vector3(goal[0], goal[1], goalHeight),
    ]);
    marker.visible = true;
    targetArrow.visible = true;
    targetArrow.setDirection(
      new THREE.Vector3(Math.cos(goalYaw ?? 0), Math.sin(goalYaw ?? 0), 0),
    );
  }
  needsRender = true;
}
function manualGoal(position, heading, turning = false) {
  if (!ready || sceneLoading) return;
  const first = !goal || !walkSelected;
  walkMode(true);
  chain = false;
  goal = [...position];
  goalYaw = wrapYaw(heading);
  goalRevision++;
  if (first) {
    filteredGoal = [...goal];
    filteredYaw = goalYaw;
    filteredHeight = goalHeight;
  }
  task = turning ? "turn" : "walk";
  setPhase(
    turning ? "turn" : "walk",
    navigationIntent === "carry" ? "carry_locomotion" : "walk",
    "Click to replace the goal; drag to set its heading. Changes apply while moving.",
  );
  syncGoal();
  if (!running) run();
}
let graspObjectChoice = 0;
let pickupCaseChoice = 0,
  pickupCases = [];
let graspObjects = [];
async function selectObject(index, initializing = false) {
  if ((sceneName === "grasp" || sceneName === "pickup") && !initializing) {
    if (
      sceneLoading ||
      index === (sceneName === "grasp" ? graspObjectChoice : pickupCaseChoice)
    )
      return;
    const resume = running;
    sceneLoading = true;
    running = false;
    requestVersion++;
    while (busy) await sleep(50);
    await scene(sceneName, index);
    const o = observations();
    neutralState = Float32Array.from(
      [...o.q, ...o.v],
      (x, i) => (x - meta.state_mean[i]) / Math.max(meta.state_std[i], 1e-6),
    );
    if (resume) run();
    return;
  }
  if (!sceneConfig.boxes?.[index]) return;
  if (holdingBox && index !== activeBox) {
    status(
      "BOX HELD",
      "Carry the held box, or reset before choosing another target.",
    );
    return;
  }
  if (running) {
    running = false;
    requestVersion++;
    $("#pause").textContent = "Run";
  }
  activeBox = index;
  const box = sceneConfig.boxes[index];
  sceneConfig.box_body = box.body;
  sceneConfig.box_qadr = box.qadr;
  sceneConfig.object_geom = objectGeom = box.geom;
  pickupSettleUntil = graspSettleUntil = 0;
  holdingLossFrames = 0;
  initialBoxHeight = boxCenter().z;
  doorHandContactStats = {
    leftFrames: 0,
    rightFrames: 0,
    leftImpulseNs: 0,
    rightImpulseNs: 0,
  };
  heldFrames = 0;
  contactWindow = [];
  maxLift = 0;
  document.querySelectorAll("[data-object]").forEach((b) => {
    b.classList.toggle(
      "active",
      Number(b.dataset.object) ===
        (sceneName === "grasp"
          ? graspObjectChoice
          : sceneName === "pickup"
            ? pickupCaseChoice
            : index),
    );
    b.setAttribute(
      "aria-pressed",
      String(
        Number(b.dataset.object) ===
          (sceneName === "grasp"
            ? graspObjectChoice
            : sceneName === "pickup"
              ? pickupCaseChoice
              : index),
      ),
    );
  });
  for (const g of geoms)
    g.mesh.material.emissive?.setHex(
      g.mesh.userData.boxIndex === index ? 0x171329 : 0,
    );
  status(
    "OBJECT SELECTED",
    `${box.label}. Choose ${sceneName === "grasp" ? "Grasp" : "Pickup"} to lift it.`,
  );
  needsRender = true;
}

let contactBuffer;
let lastPlanAudit = null;
const capturePlanAudit = new URLSearchParams(location.search).has("audit");
let samplerNoise = null,
  noisePlan = 0,
  commandNoiseRows = new Map();
const noiseCache = new Map();
let handGeoms = new Set(),
  objectGeom = -1,
  contactWindow = [],
  currentContactForce = 0,
  maxLift = 0;
function rootYaw() {
  const q = data.qpos;
  return Math.atan2(
    2 * (q[3] * q[6] + q[4] * q[5]),
    1 - 2 * (q[5] ** 2 + q[6] ** 2),
  );
}
const wrapYaw = (angle) => Math.atan2(Math.sin(angle), Math.cos(angle));
function drawCondition(commandValues) {
  const enabled =
    !!data && task !== "idle" && ["walk", "carry_locomotion"].includes(mode);
  waypointGuide.visible = enabled;
  conditionedWaypoints = [];
  if (!enabled) {
    needsRender = true;
    return;
  }
  const yaw = rootYaw(),
    cosine = Math.cos(yaw),
    sine = Math.sin(yaw),
    stride = meta.config.command_layout === "xy_yaw_height" ? 5 : 4;
  const points = [new THREE.Vector3(data.qpos[0], data.qpos[1], data.qpos[2])];
  waypointViews.forEach((view, i) => {
    const offset = i * stride,
      x = commandValues[offset],
      y = commandValues[offset + 1];
    const worldYaw =
      yaw + Math.atan2(commandValues[offset + 2], commandValues[offset + 3]);
    const point = new THREE.Vector3(
      data.qpos[0] + cosine * x - sine * y,
      data.qpos[1] + sine * x + cosine * y,
      stride === 5 ? commandValues[offset + 4] : data.qpos[2],
    );
    view.dot.position.copy(point);
    view.arrow.position.copy(point);
    view.arrow.setDirection(
      new THREE.Vector3(Math.cos(worldYaw), Math.sin(worldYaw), 0),
    );
    view.label.position
      .copy(point)
      .add(new THREE.Vector3(0, 0, 0.15 + i * 0.13));
    points.push(point);
    conditionedWaypoints.push({
      horizon: meta.horizons[i],
      x: point.x,
      y: point.y,
      z: point.z,
      yaw: wrapYaw(worldYaw),
    });
  });
  waypointPath.geometry.dispose();
  waypointPath.geometry = new THREE.BufferGeometry().setFromPoints(points);
  needsRender = true;
}
const fpv = new THREE.PerspectiveCamera(45, 426 / 240, 0.01, 50);
const depthTarget = new THREE.WebGLRenderTarget(426, 240, {
  minFilter: THREE.NearestFilter,
  magFilter: THREE.NearestFilter,
});
const depthMaterial = new THREE.MeshDepthMaterial({
  depthPacking: THREE.RGBADepthPacking,
});
const pixels = new Uint8Array(426 * 240 * 4);
let mj,
  model,
  data,
  sceneConfig,
  meta,
  actorMeta,
  actor,
  planner,
  geoms = [],
  history = [],
  lastAction = new Float32Array(29),
  lastLatent,
  selected = false,
  follow = false,
  running = false,
  ready = false,
  busy = false,
  sceneName = "pickup",
  mode = "walk",
  task = "idle",
  phase = "idle",
  goal = null,
  goalYaw = null,
  chain = false,
  simTime = 0,
  started = 0,
  heldFrames = 0,
  initialBoxHeight = 0,
  backend = "wasm",
  actorBackend = "wasm",
  requestVersion = 0,
  plannerMs = 0;
let initialState;
let lastHandTarget;
let needsRender = true;
renderer.shadowMap.autoUpdate = false;
orbit.addEventListener("change", () => {
  needsRender = true;
});
function status(skill, message) {
  $("#skill").textContent =
    task === "idle" && !running
      ? "No prompt selected"
      : `“${meta?.prompts?.[mode] || mode}”`;
  $("#status").dataset.state = skill;
  $("#status").textContent = message;
}
function resize() {
  needsRender = true;
  const w = viewport.clientWidth,
    h = viewport.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);
resize();
function pose() {
  if (!data) return;
  needsRender = true;
  renderer.shadowMap.needsUpdate = true;
  const m4 = new THREE.Matrix4();
  for (const { id, mesh } of geoms) {
    const p = id * 3,
      r = id * 9,
      a = data.geom_xmat;
    m4.set(
      a[r],
      a[r + 1],
      a[r + 2],
      data.geom_xpos[p],
      a[r + 3],
      a[r + 4],
      a[r + 5],
      data.geom_xpos[p + 1],
      a[r + 6],
      a[r + 7],
      a[r + 8],
      data.geom_xpos[p + 2],
      0,
      0,
      0,
      1,
    );
    mesh.position.setFromMatrixPosition(m4);
    mesh.quaternion.setFromRotationMatrix(m4);
  }
  const yaw = rootYaw();
  headingArrow.visible = ready;
  headingArrow.position.set(data.qpos[0], data.qpos[1], 0.035);
  headingArrow.setDirection(new THREE.Vector3(Math.cos(yaw), Math.sin(yaw), 0));
  $("#yaw-readout").textContent =
    `Heading ${((yaw * 180) / Math.PI).toFixed(0)}°${goalYaw === null ? "" : ` · target ${((goalYaw * 180) / Math.PI).toFixed(0)}°`}`;
  const id = sceneConfig.camera_id,
    p = id * 3,
    r = id * 9,
    a = data.cam_xmat;
  m4.set(
    a[r],
    a[r + 1],
    a[r + 2],
    0,
    a[r + 3],
    a[r + 4],
    a[r + 5],
    0,
    a[r + 6],
    a[r + 7],
    a[r + 8],
    0,
    0,
    0,
    0,
    1,
  );
  fpv.position.fromArray(data.cam_xpos, p);
  fpv.quaternion.setFromRotationMatrix(m4);
  fpv.fov = model.cam_fovy[id];
  fpv.updateProjectionMatrix();
}
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now(),
    dt = Math.min((now - lastAnimation) / 1000, 0.05);
  lastAnimation = now;
  heightStem.visible = marker.visible && walkSelected;
  if (ready && goal && walkSelected) {
    if (filteredGoal) {
      const a = 1 - Math.exp(-dt / 0.14);
      filteredGoal = filteredGoal.map((v, i) => v + (goal[i] - v) * a);
      filteredYaw = wrapYaw(filteredYaw + wrapYaw(goalYaw - filteredYaw) * a);
      filteredHeight += (goalHeight - filteredHeight) * a;
    }
    let turn = 0,
      forward = 0,
      side = 0;
    for (const [key, start] of heldKeys)
      if (now - start > 150) {
        if (key === "q") turn++;
        if (key === "e") turn--;
        if (key === "w") forward++;
        if (key === "s") forward--;
        if (key === "a") side++;
        if (key === "d") side--;
      }
    if (turn || forward || side) {
      const yaw = rootYaw(),
        length = Math.max(1, Math.hypot(forward, side));
      manualGoal(
        [
          goal[0] +
            (dt * 0.8 * (Math.cos(yaw) * forward - Math.sin(yaw) * side)) /
              length,
          goal[1] +
            (dt * 0.8 * (Math.sin(yaw) * forward + Math.cos(yaw) * side)) /
              length,
        ],
        turn
          ? rootYaw() + turn * Math.min(yawRate * 0.8, Math.PI * 0.8)
          : goalYaw,
        task === "turn" && !forward && !side,
      );
    }
  }
  if (data && !ready) return;
  if (data && follow) {
    const root = new THREE.Vector3(...Array.from(data.qpos.slice(0, 3)));
    root.z = 0.7;
    const delta = root.clone().sub(orbit.target);
    if (delta.lengthSq() > 1e-7) {
      camera.position.add(delta.multiplyScalar(0.07));
      orbit.target.lerp(root, 0.07);
    }
  }
  orbit.update();
  if (!needsRender) return;
  needsRender = false;
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, viewport.clientWidth, viewport.clientHeight);
  renderer.render(world, camera);
  if (data) {
    const w = Math.min(280, viewport.clientWidth * 0.3),
      h = (w * 240) / 426,
      x = viewport.clientWidth - w - 18,
      y = viewport.clientHeight - h - 54;
    renderer.setScissorTest(true);
    renderer.setScissor(x, y, w, h);
    renderer.setViewport(x, y, w, h);
    renderer.clearDepth();
    const guideVisible = waypointGuide.visible,
      headingVisible = headingArrow.visible,
      markerVisible = marker.visible,
      stemVisible = heightStem.visible,
      throwVisible = throwGuide.visible;
    throwGuide.visible = false;
    heightStem.visible =
      waypointGuide.visible =
      headingArrow.visible =
      marker.visible =
        false;
    renderer.render(world, fpv);
    waypointGuide.visible = guideVisible;
    headingArrow.visible = headingVisible;
    throwGuide.visible = throwVisible;
    marker.visible = markerVisible;
    heightStem.visible = stemVisible;
    renderer.setScissorTest(false);
  }
}
animate();
async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load ${url} (${r.status})`);
  return r.json();
}
const loadedModelFiles = {};
let runEvents = [];
let loadedModelBytes = 0;
async function buffer(url, label) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${label} is not available (${r.status}).`);
  const total = Number(r.headers.get("content-length")),
    reader = r.body.getReader();
  let count = 0;
  const parts = [];
  let lastUpdate = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    count += value.length;
    if (performance.now() - lastUpdate > 150) {
      lastUpdate = performance.now();
      $("#load-note").textContent =
        `${label} · ${(count / 1048576).toFixed(0)}${total ? " / " + (total / 1048576).toFixed(0) : ""} MiB · total ${((loadedModelBytes + count) / 1048576).toFixed(0)} MiB`;
      $("#progress").value = total ? count / total : 0;
    }
  }
  const result = new Uint8Array(count);
  let off = 0;
  for (const p of parts) {
    result.set(p, off);
    off += p.length;
  }
  loadedModelFiles[url] = count;
  loadedModelBytes += count;
  return result.buffer;
}
function mkdir(path) {
  try {
    mj.FS.mkdir(path);
  } catch (e) {
    if (!mj.FS.analyzePath(path).exists) throw e;
  }
}
async function scene(
  name,
  objectChoice = name === "pickup" ? 0 : graspObjectChoice,
) {
  let nextConfig;
  sceneLoading = true;
  document.querySelectorAll("[data-scene]").forEach((b) => (b.disabled = true));
  document.querySelectorAll("[data-task]").forEach((b) => (b.disabled = true));
  running = false;
  if (!noiseCache.has(name)) {
    const response = await fetch(`models/noise/${name}.f32`);
    if (!response.ok) throw new Error("Sampler inputs failed to load.");
    noiseCache.set(name, new Float32Array(await response.arrayBuffer()));
  }
  samplerNoise = noiseCache.get(name);
  requestVersion++;
  if (name === "pickup") {
    const layouts = (await json("models/scenes/pickup/cases.json")).layouts;
    pickupCases = [
      layouts[pickupLayout === "throw" ? "ground" : pickupLayout][2],
    ];
    objectChoice = 0;
    pickupCaseChoice = objectChoice;
    nextConfig = await json(
      `models/scenes/pickup/${pickupCases[objectChoice].config}`,
    );
  } else {
    nextConfig = await json(`models/scenes/${name}/scene.json`);
  }
  if (name === "grasp") {
    graspObjects = (await json("models/scenes/grasp/objects.json")).objects;
    graspObjectChoice = objectChoice;
    nextConfig = await json(
      `models/scenes/grasp/${new URLSearchParams(location.search).get("graspStart") === "relaxed" ? graspObjects[objectChoice].config.replace("candidate-", "relaxed-") : graspObjects[objectChoice].config}`,
    );
    if (!useTeacher && nextConfig.browser_start_xy) {
      const profile =
        navigator.vendor === "Apple Computer, Inc." ? "webkit" : "default";
      const xy = nextConfig.browser_start_xy[profile];
      if (xy) {
        nextConfig.qpos[0] = xy[0];
        nextConfig.qpos[1] = xy[1];
        nextConfig.browser_preset = `student-1hz-${profile}`;
      }
    }
    if (
      useTeacher &&
      !useF16Storage &&
      backend === "webgpu" &&
      actorBackend === "webgpu" &&
      navigator.vendor === "Apple Computer, Inc."
    ) {
      nextConfig.grasp_solver_steps = [4, 16, 8][objectChoice];
      nextConfig.browser_preset = "webkit-webgpu";
      if (objectChoice === 2) {
        nextConfig.qpos[0] = -0.06;
        nextConfig.qpos[1] = 0.6;
        nextConfig.sampler_noise = "grasp-42";
      }
    }
  }
  if (nextConfig.sampler_noise) {
    const key = nextConfig.sampler_noise;
    if (!noiseCache.has(key)) {
      const response = await fetch(`models/noise/${key}.f32`);
      if (!response.ok)
        throw new Error("Object sampler inputs failed to load.");
      noiseCache.set(key, new Float32Array(await response.arrayBuffer()));
    }
    samplerNoise = noiseCache.get(key);
  }
  for (const key of Object.values(nextConfig.command_noise || {}).concat(
    nextConfig.switch_sampler_noise || [],
  )) {
    if (!noiseCache.has(key)) {
      const response = await fetch(`models/noise/${key}.f32`);
      if (!response.ok)
        throw new Error("Command sampler inputs failed to load.");
      noiseCache.set(key, new Float32Array(await response.arrayBuffer()));
    }
  }
  activeBox = 0;
  $("#pickup-settings").hidden = name !== "pickup";
  $("#box-mode-description").textContent =
    pickupLayout === "throw"
      ? "Drag the box and release to throw. Longer drags throw farther. Right-drag to orbit."
      : "One box. Select Pickup to lift it; Carry places a destination.";
  $("#object-settings").hidden = name !== "grasp";
  mkdir("/fiber");
  mkdir(`/fiber/${name}`);
  mkdir(`/fiber/${name}/assets`);
  await Promise.all(
    nextConfig.files.map(async (file) => {
      const response = await fetch(`models/scenes/${name}/${file}`);
      if (!response.ok) throw new Error(`Missing scene asset ${file}`);
      mj.FS.writeFile(
        `/fiber/${name}/${file}`,
        new Uint8Array(await response.arrayBuffer()),
      );
    }),
  );
  const nextModel = mj.MjModel.from_xml_path(
    `/fiber/${name}/${nextConfig.xml || "scene.xml"}`,
  );
  const nextData = new mj.MjData(nextModel);
  if (data) data.delete();
  if (model) model.delete();
  sceneName = name;
  sceneConfig = nextConfig;
  model = nextModel;
  data = nextData;
  for (const { mesh } of geoms) {
    world.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  geoms = [];
  handGeoms = new Set(
    nextConfig.visuals
      .filter((g) => /(?:left|right)_(?:hand|wrist)/.test(g.name))
      .map((g) => g.id),
  );
  objectGeom =
    nextConfig.object_geom ??
    nextConfig.visuals.find((g) => g.name === "pickup_object_box").id;
  for (const g of nextConfig.visuals) {
    if (g.type === 0) continue;
    let geometry;
    const [x, y, z] = g.size;
    switch (g.type) {
      case 2:
        geometry = new THREE.SphereGeometry(x, 16, 12);
        break;
      case 3:
        geometry = new THREE.CapsuleGeometry(x, y * 2, 6, 12);
        geometry.rotateX(Math.PI / 2);
        break;
      case 4:
        geometry = new THREE.SphereGeometry(1, 16, 12);
        geometry.scale(x, y, z);
        break;
      case 5:
        geometry = new THREE.CylinderGeometry(x, x, y * 2, 16);
        geometry.rotateX(Math.PI / 2);
        break;
      case 6:
        geometry = new THREE.BoxGeometry(x * 2, y * 2, z * 2);
        break;
      case 7:
        geometry = new THREE.BufferGeometry();
        const mid = model.geom_dataid[g.id],
          va = model.mesh_vertadr[mid] * 3,
          vn = model.mesh_vertnum[mid] * 3,
          fa = model.mesh_faceadr[mid] * 3,
          fn = model.mesh_facenum[mid] * 3;
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(
            new Float32Array(model.mesh_vert.slice(va, va + vn)),
            3,
          ),
        );
        geometry.setIndex(Array.from(model.mesh_face.slice(fa, fa + fn)));
        geometry.computeVertexNormals();
        break;
      default:
        continue;
    }
    const mat = model.geom_matid[g.id],
      rgba =
        mat >= 0
          ? Array.from(model.mat_rgba.slice(mat * 4, mat * 4 + 4))
          : g.rgba;
    if (rgba[3] < 0.01) continue;
    const mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(...rgba.slice(0, 3)),
        roughness: 0.6,
        metalness: 0.16,
      }),
    );
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData.boxIndex = nextConfig.boxes
      ? nextConfig.boxes.findIndex((b) => b.body === g.body)
      : g.body === nextConfig.box_body
        ? 0
        : -1;
    mesh.userData.box = mesh.userData.boxIndex >= 0;
    world.add(mesh);
    geoms.push({ id: g.id, mesh });
  }
  const targetList = $(name === "grasp" ? "#grasp-targets" : "#box-targets");
  targetList.replaceChildren();
  for (const [index, object] of (name === "grasp"
    ? graspObjects
    : name === "pickup"
      ? pickupCases
      : nextConfig.boxes || []
  ).entries()) {
    const button = document.createElement("button");
    const shape = name === "grasp" ? ["fruit", "bottle", "cup"][index] : "box";
    const color =
      name === "grasp"
        ? ["#e9a958", "#78bac7", "#aa9ac6"][index]
        : ["#cfb795", "#88b4c8", "#ab9fc6"][index];
    const artwork =
      shape === "bottle"
        ? '<path d="M12 5h8v5l3 5v20q0 3-3 3h-8q-3 0-3-3V15l3-5z" fill="currentColor"/><path d="M12 6h8M10 20h12M10 29h12" stroke="white" stroke-opacity=".35" fill="none"/>'
        : shape === "cup"
          ? '<path d="M5 13h18v17q0 7-9 7t-9-7z" fill="currentColor"/><path d="M23 16h3q6 0 4 8-1 5-7 5" fill="none" stroke="currentColor" stroke-width="3"/><ellipse cx="14" cy="13" rx="9" ry="3" fill="#17202b" opacity=".35"/>'
          : shape === "cube" || shape === "box"
            ? '<path d="M16 8 28 15v14l-12 7-12-7V15z" fill="currentColor"/><path d="m4 15 12 7 12-7M16 22v14" fill="none" stroke="#17202b" stroke-opacity=".35"/>'
            : '<circle cx="16" cy="23" r="12" fill="currentColor"/><ellipse cx="12" cy="19" rx="4" ry="5" fill="white" opacity=".15"/>' +
              (shape === "fruit"
                ? '<path d="M16 12c0-7 7-8 9-5-3 5-7 5-9 5" fill="#8baf80"/>'
                : "");
    const title =
      name === "grasp"
        ? object.label
        : `${object.size_label || `Box ${index + 1}`} box`;
    const dimensions = object.dimensions
      ?.map((v) => Number((v * 100).toFixed(1)))
      .join(" × ");
    const iconScale = name === "pickup" ? [0.75, 0.9, 1.05][index] : 1;
    button.setAttribute(
      "aria-label",
      `${title}${dimensions ? `, ${dimensions} cm` : ""}, ${object.surface === "support" ? "on support" : name === "grasp" ? "on tabletop" : "on ground"}`,
    );
    button.innerHTML = `<svg aria-hidden="true" viewBox="0 0 32 44" style="color:${color}"><g transform="translate(16 22) scale(${iconScale}) translate(-16 -22)">${artwork}</g></svg><span class="object-name">${title}</span><span class="object-surface">${dimensions ? `${dimensions} cm · ` : ""}${name === "grasp" ? "Tabletop" : object.surface === "support" ? "On support" : "On ground"}</span>`;
    button.setAttribute("aria-pressed", "false");
    button.dataset.object = index;
    button.onclick = () => selectObject(index);
    targetList.append(button);
  }
  reset();
  if (nextConfig.boxes) selectObject(0, true);
  $(".intro").textContent =
    name === "grasp"
      ? "Choose an object to load its own scene. Each grasp starts with lowered arms and open fingers; success varies."
      : name === "pickup"
        ? "Ground, Support, or Throw. Starts vary from 0.9–1.5 m away. Pickup requests lifting directly; Carry places a destination."
        : "Choose a target and an interaction. Use Walk to place a goal.";
  document.querySelectorAll("[data-task]").forEach((button) => {
    const task = button.dataset.task;
    button.hidden = button.disabled =
      name === "grasp"
        ? task !== "grasp" && task !== "walk" && task !== "carry"
        : task === "grasp" ||
          ((task === "door" || task === "chain") && name !== "door");
  });
  window.fiberSceneDebug = {
    observations,
    depth,
    get data() {
      return data;
    },
    get model() {
      return model;
    },
    get config() {
      return nextConfig;
    },
  };
  $("[data-task=carry]").disabled = false;
  sceneLoading = false;
  document
    .querySelectorAll("[data-scene]")
    .forEach((b) => (b.disabled = false));
  document.querySelector(".fpv-label").style.display = "block";
  document
    .querySelectorAll("[data-scene]")
    .forEach((b) => b.classList.toggle("active", b.dataset.scene === name));
}
function observations() {
  const q = sceneConfig.qadr.map((i) => data.qpos[i]),
    v = sceneConfig.vadr.map((i) => data.qvel[i]);
  const quat = new THREE.Quaternion(
    data.qpos[4],
    data.qpos[5],
    data.qpos[6],
    data.qpos[3],
  );
  const gravity = new THREE.Vector3(0, 0, -1)
    .applyQuaternion(quat.invert())
    .toArray();
  const angular = Array.from(data.qvel.slice(3, 6), (x) => x * 0.25),
    deviation = q.map((x, i) => x - sceneConfig.default_q[i]);
  return {
    q,
    v,
    frame: {
      actions: Array.from(lastAction),
      ang_vel: angular,
      dof_pos: deviation,
      dof_vel: v,
      gravity,
    },
    state: new Float32Array([...deviation, ...v, ...gravity, ...angular]),
  };
}
let locomotionBank = null,
  bankGoal = null,
  bankArrived = false;
function useLocomotionBank() {
  return (
    !!locomotionBank &&
    mode === "walk" &&
    (walkSelected || task === "idle" || phase === "cross_door" ||
      phase === "approach_door" || phase === "settle_door") &&
    (!walkSelected ||
      (Math.abs(goalHeight - 0.75) < 0.01 &&
        Math.abs(filteredHeight - 0.75) < 0.01))
  );
}
let pickupSpawn = null,
  fallReported = false;
function reset(resume = running) {
  finishThrow(true);
  draggingGoal = null;
  bankGoal = null;
  bankArrived = false;
  fallReported = false;
  runEvents = [{ type: "reset", simTime: 0 }];
  running = false;
  walkMode(false);
  heldKeys.clear();
  filteredGoal = null;
  filteredYaw = null;
  requestVersion++;
  mj.mj_resetData(model, data);
  data.qpos.set(sceneConfig.qpos);
  data.qvel.set(sceneConfig.qvel);
  data.mocap_pos.set(sceneConfig.mocap_pos);
  data.mocap_quat.set(sceneConfig.mocap_quat);
  goalHeight = filteredHeight = 0.75;
  pickupSpawn = null;
  if (sceneName === "pickup" && sceneConfig.pickup_spawn) {
    const spawn = sceneConfig.pickup_spawn;
    const a = sceneConfig.box_qadr;
    const bx = data.qpos[a],
      by = data.qpos[a + 1];
    const radius =
      spawn.radius_min + Math.random() * (spawn.radius_max - spawn.radius_min);
    const side =
      ((Math.random() * 2 - 1) * spawn.sector_degrees * Math.PI) / 180;
    const bearing = Math.atan2(data.qpos[1] - by, data.qpos[0] - bx) + side;
    data.qpos[0] = bx + radius * Math.cos(bearing);
    data.qpos[1] = by + radius * Math.sin(bearing);
    const headingOffset =
      ((Math.random() * 2 - 1) *
        Math.min(20, spawn.heading_offset_degrees) *
        Math.PI) /
      180;
    const heading = bearing + Math.PI + headingOffset;
    const delta = wrapYaw(heading - rootYaw());
    const c = Math.cos(delta / 2),
      t = Math.sin(delta / 2);
    const [w, x, y, z] = Array.from(data.qpos.slice(3, 7));
    data.qpos.set(
      [c * w - t * z, c * x - t * y, c * y + t * x, c * z + t * w],
      3,
    );
    {
      // Rotate the initial box pose with the sampled approach sector.
      // This preserves the demonstrated grasp face across varied world starts.
      const [bw, bx, by, bz] = Array.from(data.qpos.slice(a + 3, a + 7));
      data.qpos.set(
        [c * bw - t * bz, c * bx - t * by, c * by + t * bx, c * bz + t * bw],
        a + 3,
      );
    }
    pickupSpawn = {
      radius,
      side,
      headingOffset,
      position: Array.from(data.qpos.slice(0, 2)),
      heading,
    };
    runEvents.push({ type: "pickup_spawn", ...pickupSpawn });
  }
  mj.mj_forward(model, data);
  lastAction = new Float32Array(29);
  lastHandTarget = sceneConfig.hand
    ? Float32Array.from(sceneConfig.hand.qadr, (a) => data.qpos[a])
    : null;
  const frame = observations().frame;
  const zeroFrame = Object.fromEntries(
    Object.entries(frame).map(([key, values]) => [key, values.map(() => 0)]),
  );
  history = Array.from({ length: actorMeta.history_length + 1 }, () =>
    structuredClone(zeroFrame),
  );
  history[0] = structuredClone(frame);
  lastLatent = new Float32Array(sceneConfig.anchor);
  simTime = 0;
  noisePlan = 0;
  commandNoiseRows.clear();
  graspSwitchSamplerActive = false;
  carryRouteStage = 0;
  lastExecutedCommand = null;
  stateDropPlans = 0;
  task = phase = "idle";
  mode = "walk";
  goal = null;
  goalYaw = null;
  chain = false;
  selected = false;
  heldFrames = 0;
  holdingBox = false;
  navigationIntent = "walk";
  $("[data-task=carry]").disabled = false;
  contactWindow = [];
  currentContactForce = maxLift = 0;
  marker.visible = false;
  waypointGuide.visible = false;
  conditionedWaypoints = [];
  initialBoxHeight = boxCenter().z;
  doorHandContactStats = {
    leftFrames: 0,
    rightFrames: 0,
    leftImpulseNs: 0,
    rightImpulseNs: 0,
  };
  initialState = observations().state;
  $("#pause").textContent = "Run";
  $("#sim-time").textContent = "0.0 s";
  $("#speed").textContent = "—";
  status(
    "STANDING",
    sceneName === "grasp"
      ? "Choose Grasp object to lift the original tabletop object with articulated Dex3 hands."
      : "Select the box or choose a task. Drag to explore the scene.",
  );
  document
    .querySelectorAll("[data-task]")
    .forEach((b) => b.classList.remove("active"));
  camera.position.set(
    sceneName === "door" ? -3.5 : -2.3,
    sceneName === "door" ? -5 : sceneName === "grasp" ? 2.4 : -3.2,
    sceneName === "door" ? 3.4 : 2.4,
  );
  orbit.target.set(
    sceneName === "door" ? 1.5 : sceneName === "grasp" ? 0 : 0.7,
    0,
    0.65,
  );
  pose();
  if (resume && ready && !sceneLoading) run();
}
function boxCenter() {
  const g = sceneConfig.visuals.find((v) =>
      sceneConfig.object_geom !== undefined
        ? v.id === sceneConfig.object_geom
        : v.body === sceneConfig.box_body && v.type === 6,
    ),
    p = g.id * 3;
  return new THREE.Vector3(
    data.geom_xpos[p],
    data.geom_xpos[p + 1],
    data.geom_xpos[p + 2],
  );
}
function handContactForce() {
  let force = 0;
  const doorForces = { left: 0, right: 0 };
  const contacts = data.contact;
  try {
    for (let i = 0; i < data.ncon; i++) {
      const contact = contacts.get(i);
      try {
        const touchesHand =
          (contact.geom1 === objectGeom && handGeoms.has(contact.geom2)) ||
          (contact.geom2 === objectGeom && handGeoms.has(contact.geom1));
        if (
          capturePlanAudit &&
          sceneName === "door" &&
          mode === "open_door" &&
          contact.efc_address >= 0
        ) {
          const handle = sceneConfig.visuals.find(
            (g) => g.name === "door_handle",
          ).id;
          const other =
            contact.geom1 === handle
              ? contact.geom2
              : contact.geom2 === handle
                ? contact.geom1
                : -1;
          const name =
            sceneConfig.visuals.find((g) => g.id === other)?.name || "";
          const side = /left_(?:hand|wrist)/.test(name)
            ? "left"
            : /right_(?:hand|wrist)/.test(name)
              ? "right"
              : null;
          if (side) {
            mj.mj_contactForce(model, data, i, contactBuffer);
            doorForces[side] += Math.max(0, contactBuffer.GetView()[0]);
          }
        }
        if (touchesHand && contact.efc_address >= 0) {
          mj.mj_contactForce(model, data, i, contactBuffer);
          force += Math.max(0, contactBuffer.GetView()[0]);
        }
      } finally {
        contact.delete();
      }
    }
  } finally {
    contacts.delete();
  }
  for (const side of ["left", "right"]) {
    if (doorForces[side] > 0.1) doorHandContactStats[`${side}Frames`]++;
    doorHandContactStats[`${side}ImpulseNs`] += doorForces[side] * 0.02;
  }
  return force;
}
function tensor(values, shape, type = "float32") {
  return new ort.Tensor(
    type,
    type === "bool" ? Uint8Array.from(values) : Float32Array.from(values),
    shape,
  );
}
function project(a, chunk = actorMeta.z_dim) {
  const result = new Float32Array(a.length);
  for (let o = 0; o < a.length; o += chunk) {
    let norm = 0;
    for (let j = 0; j < chunk; j++) norm += a[o + j] ** 2;
    const scale = Math.sqrt(chunk) / Math.max(Math.sqrt(norm), 1e-8);
    for (let j = 0; j < chunk; j++) result[o + j] = a[o + j] * scale;
  }
  return result;
}
function slerp(a, b, t) {
  let dot = 0;
  for (let j = 0; j < a.length; j++) dot += a[j] * b[j];
  const theta = Math.acos(Math.max(-1, Math.min(1, dot / a.length)));
  if (theta < 1e-4) return project(a.map((v, j) => v * (1 - t) + b[j] * t));
  const s = Math.sin(theta);
  return Float32Array.from(
    a,
    (v, j) => (Math.sin((1 - t) * theta) * v + Math.sin(t * theta) * b[j]) / s,
  );
}
function schedule(knots, anchor = lastLatent) {
  const c = meta.config,
    P = c.knot_control_points,
    Z = actorMeta.z_dim,
    cpk = meta.controls_per_knot;
  let prev = anchor;
  const plan = [];
  for (let k = 0; k < c.num_knots; k++) {
    const points = [prev];
    for (let p = 0; p < P; p++)
      points.push(knots.slice((k * P + p) * Z, (k * P + p + 1) * Z));
    for (let i = 1; i <= cpk; i++) {
      const x = (i / cpk) * P,
        seg = Math.max(0, Math.ceil(x) - 1);
      plan.push(slerp(points[seg], points[seg + 1], x - seg));
    }
    prev = points[P];
  }
  return plan;
}
function depth() {
  pose();
  const throwVisible = throwGuide.visible;
  throwGuide.visible = false;
  const markerVisible = marker.visible,
    stemVisible = heightStem.visible;
  marker.visible = heightStem.visible = false;
  const guideVisible = waypointGuide.visible,
    headingVisible = headingArrow.visible;
  waypointGuide.visible = headingArrow.visible = false;
  grid.visible = false;
  const old = world.overrideMaterial,
    background = world.background,
    fog = world.fog;
  world.overrideMaterial = depthMaterial;
  world.background = new THREE.Color(0xffffff);
  world.fog = null;
  renderer.setScissorTest(false);
  renderer.setRenderTarget(depthTarget);
  renderer.setViewport(0, 0, 426, 240);
  renderer.render(world, fpv);
  renderer.readRenderTargetPixels(depthTarget, 0, 0, 426, 240, pixels);
  renderer.setRenderTarget(null);
  world.overrideMaterial = old;
  world.background = background;
  world.fog = fog;
  grid.visible = true;
  waypointGuide.visible = guideVisible;
  headingArrow.visible = headingVisible;
  throwGuide.visible = throwVisible;
  marker.visible = markerVisible;
  heightStem.visible = stemVisible;
  const c = meta.config,
    result = new Float32Array(c.depth_height * c.depth_width);
  for (let y = 0; y < c.depth_height; y++)
    for (let x = 0; x < c.depth_width; x++) {
      const sx = Math.floor((x * 426) / c.depth_width),
        sy = 239 - Math.floor((y * 240) / c.depth_height),
        i = (sy * 426 + sx) * 4;
      const d =
        (pixels[i] / 16777216 +
          pixels[i + 1] / 65536 +
          pixels[i + 2] / 256 +
          pixels[i + 3]) /
        256;
      const z = (fpv.near * fpv.far) / (fpv.far - d * (fpv.far - fpv.near));
      const mm = Math.round(z * 1000) / 1000;
      result[y * c.depth_width + x] =
        mm >= c.min_depth_m && mm <= c.max_depth_m ? mm : 0;
    }
  return result;
}
function command() {
  const c = meta.config,
    result = [];
  const target = (walkSelected && filteredGoal) ||
      goal || [data.qpos[0], data.qpos[1]],
    dx = target[0] - data.qpos[0],
    dy = target[1] - data.qpos[1],
    q = Array.from(data.qpos.slice(3, 7)),
    yaw = Math.atan2(
      2 * (q[0] * q[3] + q[1] * q[2]),
      1 - 2 * (q[2] ** 2 + q[3] ** 2),
    ),
    dist = Math.hypot(dx, dy),
    bearing = Math.atan2(dy, dx) - yaw,
    err = Math.atan2(Math.sin(bearing), Math.cos(bearing));
  for (const h of meta.horizons) {
    const progress = Math.min(
        task === "turn" ? 0 : dist,
        (mode === "carry_locomotion" ? 1 : 0.8) * h,
      ),
      yawError =
        phase === "approach_object" && dist > 0.35
          ? err
          : goalYaw === null
            ? dist > 1e-5
              ? err
              : 0
            : wrapYaw(
                (walkSelected ? (filteredYaw ?? goalYaw) : goalYaw) - yaw,
              ),
      rate = walkSelected ? yawRate : 0.9,
      turn = Math.max(-rate * h, Math.min(rate * h, yawError));
    result.push(
      progress * Math.cos(err),
      progress * Math.sin(err),
      Math.sin(turn),
      Math.cos(turn),
    );
    if (c.command_layout === "xy_yaw_height")
      result.push(walkSelected ? filteredHeight : 0.75);
  }
  return new Float32Array(result);
}
async function plan(version) {
  const noiseKey =
    sceneConfig.command_noise?.[mode] ||
    (mode === "grasp_table" && graspSwitchSamplerActive
      ? sceneConfig.switch_sampler_noise
      : null);
  const planNoise = noiseKey ? noiseCache.get(noiseKey) : samplerNoise;
  const requestSimTime = simTime,
    requestNoiseRow = noiseKey
      ? commandNoiseRows.get(noiseKey) || 0
      : noisePlan;
  const anchor = Float32Array.from(lastLatent);
  const c = meta.config,
    obs = observations(),
    state = Float32Array.from(
      [...obs.q, ...obs.v],
      (x, i) => (x - meta.state_mean[i]) / Math.max(meta.state_std[i], 1e-6),
    ),
    d = depth(),
    cmd = command();
  drawCondition(cmd);
  const commandForPlan = mode;
  const dropStateForPlan = shouldDropState(commandForPlan);
  let trajectory = Float32Array.from({ length: c.num_knots * c.z_dim }, () => {
    const u = Math.max(Math.random(), 1e-8);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  });
  const handConfig = sceneConfig.hand;
  const activeHand = !!handConfig && mode === "grasp_table";
  const handState = activeHand
    ? Float32Array.from(
        handConfig.qadr,
        (a, i) =>
          (data.qpos[a] - meta.hand_state_normalization.mean[i]) /
          Math.max(meta.hand_state_normalization.std[i], 1e-6),
      )
    : new Float32Array(c.hand_state_dim);
  const value = (data, dims, type = "float32") => ({
    type,
    data: type === "bool" ? Uint8Array.from(data) : Float32Array.from(data),
    dims,
  });
  const feed = {
    depth: value(d, [1, 1, c.depth_height, c.depth_width]),
    state: value(state, [1, c.state_dim]),
    command: value(cmd, [1, c.command_dim]),
    language: value(meta.language[mode], [1, c.language_dim]),
    drop_command: value(
      [!(mode === "walk" || mode === "carry_locomotion")],
      [1],
      "bool",
    ),
    drop_state: value([dropStateForPlan], [1], "bool"),
    hand_state: value(handState, [1, c.hand_state_dim]),
    hand_mask: value([activeHand], [1], "bool"),
  };
  const handTrajectory = Float32Array.from(
    { length: c.num_knots * c.hand_dim },
    () => {
      const u = Math.max(Math.random(), 1e-8);
      return (
        Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random())
      );
    },
  );
  if (planNoise) {
    const rowWidth = trajectory.length + handTrajectory.length;
    const offset = (requestNoiseRow % (planNoise.length / rowWidth)) * rowWidth;
    noisePlan++;
    if (noiseKey) commandNoiseRows.set(noiseKey, requestNoiseRow + 1);
    trajectory = planNoise.slice(offset, offset + trajectory.length);
    handTrajectory.set(
      planNoise.subarray(offset + trajectory.length, offset + rowWidth),
    );
  }
  const stepsForPlan =
    stepOverride ??
    (mode === "grasp_table" ? sceneConfig.grasp_solver_steps : undefined) ??
    (useTeacher
      ? walkSelected && task !== "turn" && mode === "walk"
        ? 4
        : 8
      : 3);
  if (capturePlanAudit)
    lastPlanAudit = {
      input: structuredClone({
        feed,
        trajectory,
        handTrajectory,
        steps: stepsForPlan,
      }),
      qpos: Array.from(data.qpos),
      qvel: Array.from(data.qvel),
      ctrl: Array.from(data.ctrl),
      anchor: Array.from(anchor),
      controller: structuredClone({
        qadr: sceneConfig.qadr,
        vadr: sceneConfig.vadr,
        actuator: sceneConfig.actuator,
        kp: sceneConfig.kp,
        kd: sceneConfig.kd,
        effort: sceneConfig.effort,
        hand: sceneConfig.hand,
      }),
      physicsOptions: Object.fromEntries(
        [
          "timestep",
          "integrator",
          "cone",
          "jacobian",
          "solver",
          "iterations",
          "ls_iterations",
          "disableflags",
          "enableflags",
          "tolerance",
          "impratio",
        ].map((k) => [k, model.opt[k]]),
      ),
      warmstart: Array.from(data.qacc_warmstart),
      modelArrays: Object.fromEntries(
        [
          "body_mass",
          "body_inertia",
          "body_ipos",
          "dof_armature",
          "dof_damping",
          "geom_friction",
          "geom_condim",
          "geom_solref",
          "geom_solimp",
          "actuator_gainprm",
          "actuator_biasprm",
          "geom_pos",
          "geom_quat",
          "actuator_gear",
          "actuator_ctrlrange",
        ].map((k) => [k, Array.from(model[k])]),
      ),
    };
  const t = performance.now();
  const result = await planner.sample({
    feed,
    trajectory,
    handTrajectory,
    trajectoryDims: [1, c.num_knots, c.z_dim],
    handDims: [1, c.num_knots, c.hand_dim],
    steps: stepsForPlan,
  });
  const elapsedMs = performance.now() - t;
  if (version !== requestVersion) return [];
  plannerMs = elapsedMs;
  $("#planning").textContent =
    `Plan ${(plannerMs / 1000).toFixed(2)} s · ${stepsForPlan} steps · ${backend.toUpperCase()}`;
  const controls = schedule(project(result.trajectory), anchor);
  if (capturePlanAudit) {
    lastPlanAudit.trajectory = Array.from(result.trajectory);
    lastPlanAudit.hand = Array.from(result.hand);
    lastPlanAudit.firstZ = Array.from(controls[0]);
  }
  if (dropStateForPlan) stateDropPlans++;
  runEvents.push({
    type: "plan",
    simTime,
    requestSimTime,
    command: commandForPlan,
    droppedState: dropStateForPlan,
    noiseRow: requestNoiseRow,
    sampler: noiseKey || sceneConfig.sampler_noise || sceneName,
    milliseconds: elapsedMs,
    conditionMs: result.conditionMs,
    solverMs: result.elapsedMs - result.conditionMs,
    wasmThreads: result.wasmThreads,
    solverSteps: stepsForPlan,
    controlsPerPlan,
  });
  return controls.map((z, i) => ({
    z,
    command: commandForPlan,
    hand: activeHand
      ? Float32Array.from(
          result.hand.slice(
            Math.floor(i / meta.controls_per_knot) * c.hand_dim,
            (Math.floor(i / meta.controls_per_knot) + 1) * c.hand_dim,
          ),
          (v, j) =>
            meta.hand_action.center[j] +
            Math.max(-1, Math.min(1, v)) * meta.hand_action.scale[j],
        )
      : null,
  }));
}

async function inferActor(z) {
  const obs = observations(),
    hist = [];
  for (const key of ["actions", "ang_vel", "dof_pos", "dof_vel", "gravity"])
    for (const fr of history.slice(1)) hist.push(...fr[key]);
  const feed = {
    state: tensor(obs.state, [1, 64]),
    last_action: tensor(lastAction, [1, 29]),
    history_actor: tensor(hist, [1, hist.length]),
    z: tensor(z, [1, actorMeta.z_dim]),
  };
  let action;
  try {
    const output = await actor.run(feed);
    action = Float32Array.from(output.action.data);
    Object.values(output).forEach((v) => v.dispose());
  } finally {
    Object.values(feed).forEach((v) => v.dispose());
  }
  return action;
}
async function control({ z, hand, command }, version) {
  const action = await inferActor(z);
  if (version !== requestVersion) return;
  if (!action.every(Number.isFinite))
    throw new Error(
      "Controller produced invalid actions. Reset before retrying.",
    );
  if (hand) lastHandTarget = hand;
  lastAction = Float32Array.from(action, (v) =>
    Math.max(-6, Math.min(6, v * 6)),
  );
  const target = Array.from(
    lastAction,
    (a, i) => sceneConfig.default_q[i] + a * 0.25 * actorMeta.action_rescale[i],
  );
  const auditFirstControl =
    capturePlanAudit && lastPlanAudit && !lastPlanAudit.firstAction;
  if (auditFirstControl) lastPlanAudit.physicsSteps = [];
  for (let step = 0; step < sceneConfig.decimation; step++) {
    mj.mj_step1(model, data);
    for (let i = 0; i < 29; i++) {
      const torque =
        sceneConfig.kp[i] * (target[i] - data.qpos[sceneConfig.qadr[i]]) -
        sceneConfig.kd[i] * data.qvel[sceneConfig.vadr[i]];
      data.ctrl[sceneConfig.actuator[i]] = Math.max(
        -sceneConfig.effort[i],
        Math.min(sceneConfig.effort[i], torque),
      );
    }
    if (lastHandTarget && sceneConfig.hand) {
      const h = sceneConfig.hand;
      for (let i = 0; i < h.qadr.length; i++) {
        const torque =
          h.kp[i] * (lastHandTarget[i] - data.qpos[h.qadr[i]]) -
          h.kd[i] * data.qvel[h.vadr[i]];
        data.ctrl[h.actuator[i]] = Math.max(
          -h.effort[i],
          Math.min(h.effort[i], torque),
        );
      }
    }
    const auditCtrl = auditFirstControl ? Array.from(data.ctrl) : null;
    mj.mj_step2(model, data);
    if (auditFirstControl)
      lastPlanAudit.physicsSteps.push({
        ctrl: auditCtrl,
        qpos: Array.from(data.qpos),
        qvel: Array.from(data.qvel),
      });
  }
  if (capturePlanAudit && lastPlanAudit && !lastPlanAudit.firstAction) {
    lastPlanAudit.firstAction = Array.from(action);
    lastPlanAudit.firstHandTarget = lastHandTarget
      ? Array.from(lastHandTarget)
      : null;
    lastPlanAudit.actorRescale = actorMeta.action_rescale;
    lastPlanAudit.firstQpos = Array.from(data.qpos);
  }
  if (command !== null) lastExecutedCommand = command;
  lastLatent = Float32Array.from(z);
  history.unshift(observations().frame);
  history.length = actorMeta.history_length + 1;
  simTime += 0.02;
  pose();
  $("#sim-time").textContent = simTime.toFixed(1) + " s";
  if (!Number.isFinite(data.qpos[2]))
    throw new Error("Simulation produced a non-finite state.");
  if (data.qpos[2] < 0.25 && !fallReported) {
    fallReported = true;
    status(
      "FALLEN",
      "The policy lost balance. Simulation continues; reset to try again.",
    );
  }
  arbiter();
}
let phaseStart = 0,
  neutralState;
function setPhase(p, m, message) {
  runEvents.push({
    type: "phase",
    phase: p,
    command: m,
    simTime,
    root: Array.from(data.qpos.slice(0, 7)),
  });
  phase = p;
  mode = m;
  phaseStart = simTime;
  status(p.replaceAll("_", " ").toUpperCase(), message);
}
function arbiter() {
  if (task === "idle") return;
  if (
    phase === "turn" &&
    !heldKeys.size &&
    goalYaw !== null &&
    Math.abs(wrapYaw(goalYaw - rootYaw())) < 0.08 &&
    Math.abs(data.qpos[2] - goalHeight) < 0.08
  ) {
    phase = "heading_reached";
    status(
      "COMPLETE",
      "Heading reached. Choose a prompt or set another waypoint.",
    );
    pose();
    return;
  }
  if (
    phase === "approach_object" &&
    goal &&
    Math.hypot(data.qpos[0] - goal[0], data.qpos[1] - goal[1]) <
      (sceneName === "pickup" ? 0.1 : 0.18) &&
    Math.abs(wrapYaw(goalYaw - rootYaw())) <
      (sceneName === "pickup" ? 0.25 : 0.35)
  ) {
    goal = null;
    goalYaw = null;
    setPhase(
      sceneName === "grasp" ? "grasp" : "pickup",
      sceneName === "grasp" ? "grasp_table" : "pickup",
      "Lift the selected object.",
    );
  }
  if (phase === "settle_grasp" && simTime >= graspSettleUntil) {
    setPhase(
      "grasp",
      "grasp_table",
      "FIBER plans the body motion and Dex3 finger targets to lift the object.",
    );
  }
  if (phase === "settle_pickup" && simTime >= pickupSettleUntil) {
    setPhase("pickup", "pickup", "FIBER is searching for and lifting the box.");
  }
  const box = boxCenter(),
    root = new THREE.Vector3(data.qpos[0], data.qpos[1], data.qpos[2]),
    raised =
      box.z > initialBoxHeight + 0.1 &&
      root.z > 0.25 &&
      Math.hypot(box.x - root.x, box.y - root.y) < 0.8;
  maxLift = Math.max(maxLift, box.z - initialBoxHeight);
  currentContactForce = handContactForce();
  if (holdingBox) {
    holdingLossFrames =
      raised && currentContactForce > 0.1 ? 0 : holdingLossFrames + 1;
    if (holdingLossFrames >= 25) {
      holdingBox = false;
      runEvents.push({ type: "box_contact_lost", simTime, phase });
    }
  }
  contactWindow.push(raised && currentContactForce > 0.1);
  if (contactWindow.length > 51) contactWindow.shift();
  heldFrames = raised ? heldFrames + 1 : 0;
  const heldWithContact =
    heldFrames >= 51 && contactWindow.filter(Boolean).length >= 45;
  if (
    phase === "walk" &&
    goal &&
    Math.hypot(root.x - goal[0], root.y - goal[1]) <
      (useLocomotionBank() ? (bankArrived ? 0.5 : 0.3) : 0.2) &&
    (!walkSelected || Math.abs(root.z - goalHeight) < 0.08) &&
    (goalYaw === null || Math.abs(wrapYaw(goalYaw - rootYaw())) < 0.12)
  ) {
    setPhase(
      "goal_reached",
      mode,
      "Walking goal reached. Click to place another goal.",
    );
  }
  if (
    phase === "approach_door" &&
    goal && Math.hypot(root.x - goal[0], root.y - goal[1]) < 0.2 &&
    Math.abs(wrapYaw(goalYaw - rootYaw())) < 0.16
  ) {
    goal = null;
    goalYaw = 0;
    marker.visible = false;
    setPhase("settle_door", "walk", "Settling the stance before opening the door.");
  }
  if (phase === "settle_door" && simTime - phaseStart >= 1) {
    goal = null;
    goalYaw = null;
    setPhase("open_door", "open_door", "FIBER is interacting with the door.");
  }
  if (
    phase === "open_door" &&
    data.qpos[sceneConfig.hinge_qadr] > 0.95 &&
    root.x > 1.1
  ) {
    goal = [2.3, -0.15];
    goalYaw = 0;
    bankGoal = null;
    marker.position.set(...goal, 0.008);
    marker.visible = true;
    setPhase("cross_door", "walk", "Walk through the opened doorway.");
  }
  if (
    phase === "cross_door" &&
    root.x > 2.1 &&
    Math.abs(root.y) < 0.3 &&
    Math.hypot(root.x - goal[0], root.y - goal[1]) < 0.3 &&
    Math.abs(wrapYaw(rootYaw())) < 0.2
  ) {
    goal = null;
    goalYaw = null;
    marker.visible = false;
    if (chain) {
      pickupSettleUntil = simTime + (sceneConfig.pickup_settle_seconds ?? 3);
      setPhase(
        "settle_pickup",
        "walk",
        "Settling the stance before lifting the box.",
      );
    } else {
      task = "idle";
      setPhase(
        "complete",
        "walk",
        "Door opened and crossed. Choose another task.",
      );
    }
  }
  if (phase === "grasp" && heldWithContact) {
    setPhase(
      "holding_object",
      "grasp_table",
      "Object lifted and held with finger contact for one second. Reset to try again.",
    );
  }
  if (phase === "pickup" && heldWithContact) {
    holdingBox = true;
    $("[data-task=carry]").disabled = false;
    $("[data-task=carry]").title =
      "Click to place a goal; drag to choose the heading";
    if (task === "pickup") {
      goal = null;
      setPhase(
        "holding",
        "pickup",
        "Box lifted and held for one second. Select Carry, then click and drag a floor waypoint.",
      );
    } else if (chain) {
      goal = [2.3, -0.1];
      goalYaw = Math.PI;
      carryRouteStage = 0;
      marker.position.set(...goal, 0.008);
      marker.visible = true;
      setPhase(
        "carry",
        "carry_locomotion",
        "Carry the box back through the doorway.",
      );
    } else if (goal) {
      setPhase(
        "carry",
        "carry_locomotion",
        "Carry toward the selected floor goal.",
      );
    }
  }
  if (
    chain &&
    phase === "carry" &&
    carryRouteStage === 0 &&
    root.x < 2.65 &&
    Math.abs(wrapYaw(Math.PI - rootYaw())) < 0.4
  ) {
    carryRouteStage = 1;
    goal = [0.3, 0];
    marker.position.set(...goal, 0.008);
  }
  if (
    phase === "carry" &&
    heldWithContact &&
    (!chain || (carryRouteStage === 1 && box.x < 1.1)) &&
    goal &&
    Math.hypot(root.x - goal[0], root.y - goal[1]) < 0.7
  ) {
    setPhase(
      "complete",
      "carry_locomotion",
      "Box carried to the destination. Choose another goal or reset.",
    );
  }
  if (
    phase === "put_down" &&
    simTime - phaseStart > 2 &&
    box.z < initialBoxHeight + 0.06
  ) {
    task = "idle";
    setPhase("complete", "walk", "Interaction finished. Choose another goal.");
  }
}
async function loop() {
  if (busy) return;
  busy = true;
  const version = requestVersion;
  try {
    while (running && version === requestVersion) {
      const revision = goalRevision;
      const bankControl =
        useLocomotionBank() ||
        phase === "settle_pickup" ||
        phase === "settle_grasp";
      let controls;
      if (bankControl) {
        const startedControl = performance.now();
        const liveGoal = (walkSelected && filteredGoal) || goal;
        if (
          !goal ||
          !bankGoal ||
          Math.hypot(goal[0] - bankGoal[0], goal[1] - bankGoal[1]) > 0.01
        ) {
          bankGoal = goal ? [...goal] : null;
          bankArrived = false;
        }
        const distance = liveGoal
          ? Math.hypot(liveGoal[0] - data.qpos[0], liveGoal[1] - data.qpos[1])
          : 0;
        const arrivalRadius = phase === "approach_door" ? 0.16 : 0.3;
        bankArrived = distance <= arrivalRadius || (bankArrived && distance < arrivalRadius + (phase === "approach_door" ? 0.04 : 0.2));
        const targetZ =
          phase === "settle_grasp"
            ? Float32Array.from(sceneConfig.anchor)
            : locomotionZ(locomotionBank, {
                x: data.qpos[0],
                y: data.qpos[1],
                yaw: rootYaw(),
                goal: (walkSelected && filteredGoal) || goal,
                heading: walkSelected ? (filteredYaw ?? goalYaw) : goalYaw,
                yawRate,
                arrived: bankArrived,
                arrivalRadius,
                vx: data.qvel[0],
                vy: data.qvel[1],
                wz: data.qvel[5],
              });
        const z = smoothLocomotionZ(lastLatent, targetZ, locomotionBank.stand);
        controls = [{ z, hand: null, command: "walk" }];
        plannerMs = performance.now() - startedControl;
        $("#planning").textContent = "Walking · direct control";
        // Direct navigation uses the live goal, not planner horizon tokens.
        waypointGuide.visible = false;
        const navigationGoal = (walkSelected && filteredGoal) || goal;
        conditionedWaypoints = navigationGoal
          ? [
              {
                kind: "navigation_goal",
                x: navigationGoal[0],
                y: navigationGoal[1],
                z: goalHeight,
                yaw: filteredYaw ?? goalYaw ?? rootYaw(),
              },
            ]
          : [];
        needsRender = true;
      } else controls = await plan(version);
      let executed = 0;
      for (const next of controls.slice(0, controlsPerPlan)) {
        if (!running || version !== requestVersion) break;
        const before = performance.now();
        await control(next, version);
        executed++;
        if (next.command !== mode && phase !== "settle_pickup") break;
        if (walkSelected && executed >= 5 && goalRevision !== revision) break;
        await sleep(Math.max(0, 20 - (performance.now() - before)));
      }
      $("#speed").textContent =
        (simTime / Math.max((performance.now() - started) / 1000, 0.1)).toFixed(
          2,
        ) + "×";
    }
  } catch (error) {
    console.error(error);
    running = false;
    status("SIMULATION ERROR", error.message);
    $("#pause").textContent = "Run";
  } finally {
    busy = false;
    if (running) setTimeout(loop, 0);
  }
}
function run() {
  if (!ready || sceneLoading) return;
  runEvents.push({ type: "run", simTime, command: mode });
  running = true;
  started = performance.now() - simTime * 1000;
  $("#pause").textContent = "Pause";
  loop();
}
function beginDoor() {
  // Return to a forward approach stance through real locomotion when a previous
  // walk changed the contact geometry. No reset or waypoint-conditioned Door.
  const stance = [0.15, -0.6];
  const displaced = Math.hypot(data.qpos[0] - stance[0], data.qpos[1] - stance[1]) > 0.25;
  const moving = Math.hypot(data.qvel[0], data.qvel[1]) > 0.12;
  if (data.qpos[0] < 1.1 && (displaced || moving || Math.abs(wrapYaw(rootYaw())) > 0.4)) {
    goal = stance;
    goalYaw = 0;
    bankGoal = null;
    bankArrived = false;
    setPhase("approach_door", "walk", "Approaching the door with the handle on the left.");
  } else {
    setPhase("open_door", "open_door", "FIBER is interacting with the door.");
  }
}
function choose(which) {
  if (!ready || sceneLoading) return;
  runEvents.push({
    type: "command",
    simTime,
    requested: which,
    previous: mode,
  });
  if (which === "walk" || which === "carry") {
    navigationIntent = which;
    requestVersion++;
    walkMode(true);
    goal = null;
    goalYaw = null;
    filteredGoal = null;
    task = "idle";
    mode = which === "carry" ? "carry_locomotion" : "walk";
    bankGoal = null;
    bankArrived = false;
    if (which === "walk") goalHeight = filteredHeight = 0.75;
    marker.visible = waypointGuide.visible = false;
    status(
      "PLACE A WAYPOINT",
      "Left-click the floor; drag from that point to set the heading.",
    );
    syncGoal();
    run();
    return;
  }
  walkMode(false);
  requestVersion++;
  chain = false;
  task = which;
  goal = null;
  goalYaw = null;
  waypointGuide.visible = false;
  marker.visible = false;
  if (which === "grasp") {
    if (sceneName !== "grasp") {
      status(
        "CHOOSE GRASP",
        "Select the Grasp scene to use articulated Dex3 hands.",
      );
      return;
    }
    if (
      simTime > 0 &&
      lastExecutedCommand !== null &&
      lastExecutedCommand !== "grasp_table"
    ) {
      graspSwitchSamplerActive = true;
      graspSettleUntil = simTime + (sceneConfig.grasp_settle_seconds ?? 2);
      setPhase(
        "settle_grasp",
        "walk",
        "Settling the resting arm pose before grasping.",
      );
    } else
      setPhase(
        "grasp",
        "grasp_table",
        "FIBER plans the body motion and Dex3 finger targets to lift the object.",
      );
  } else if (which === "chain") {
    if (sceneName !== "door") {
      status(
        "CHOOSE DOOR & BOX",
        "Select the Door & box scene to start this chain.",
      );
      return;
    }
    chain = true;
    beginDoor();
  } else if (which === "door") {
    if (sceneName !== "door") {
      status(
        "CHOOSE DOOR & BOX",
        "Select the Door & box scene to try opening the door.",
      );
      return;
    }
    beginDoor();
  } else if (which === "putdown") {
    setPhase(
      "put_down",
      "put_down_ground",
      "Set the box down near the current position.",
    );
  } else {
    if (
      sceneName === "door" &&
      !holdingBox &&
      data.qpos[sceneConfig.hinge_qadr] > 0.5
    ) {
      pickupSettleUntil = simTime + (sceneConfig.pickup_settle_seconds || 0);
    } else pickupSettleUntil = simTime;
    setPhase(
      pickupSettleUntil > simTime ? "settle_pickup" : "pickup",
      pickupSettleUntil > simTime ? "walk" : "pickup",
      pickupSettleUntil > simTime
        ? "Settling the stance before lifting the box."
        : "FIBER is searching for and lifting the box.",
    );
  }
  document
    .querySelectorAll("[data-task]")
    .forEach((b) => b.classList.toggle("active", b.dataset.task === which));
  run();
}
async function launch() {
  if (ready) return;
  $("#launch").disabled = true;
  $("#progress").hidden = false;
  try {
    [meta, actorMeta] = await Promise.all([
      json(`models/${plannerModel}/planner_meta.json`),
      json("models/actor_meta.json"),
    ]);
    $("#load-note").textContent = "Loading MuJoCo and scene geometry…";
    mj = await loadMujoco();
    contactBuffer = new mj.DoubleBuffer(6);
    await scene(sceneName);
    const obs = observations();
    neutralState = Float32Array.from(
      [...obs.q, ...obs.v],
      (x, i) => (x - meta.state_mean[i]) / Math.max(meta.state_std[i], 1e-6),
    );
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = new URL("./vendor/", location.href).href;
    const backendPreference = new URLSearchParams(location.search).get(
      "backend",
    );
    // CPU planner execution tracks the native W8 result more closely for
    // small-object grasping. Keep the FP32 actor on GPU when available.
    backend =
      backendPreference === "webgpu" && navigator.gpu
        ? "webgpu"
        : backendPreference === "wasm" || !useTeacher
          ? "wasm"
          : navigator.gpu
            ? "webgpu"
            : "wasm";
    const options = {
      executionProviders: backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
      graphOptimizationLevel: "all",
    };
    const actorBuffer = await buffer(
      "models/actor.onnx",
      "Loading FIBER controller",
    );
    const actorPreference = new URLSearchParams(location.search).get("actor");
    actorBackend =
      actorPreference === "wasm"
        ? "wasm"
        : actorPreference === "webgpu" && navigator.gpu
          ? "webgpu"
          : backendPreference === "wasm"
            ? "wasm"
            : navigator.gpu
              ? "webgpu"
              : "wasm";
    actor = await ort.InferenceSession.create(actorBuffer, {
      ...options,
      executionProviders:
        actorBackend === "wasm" ? ["wasm"] : ["webgpu", "wasm"],
    });
    $("#load-note").textContent = "Preparing the FP32 controller…";
    await inferActor(lastLatent); // Compile/warm kernels without stepping physics.
    locomotionBank = validateBank(
      await json("models/locomotion/z_bank.json"),
      actorMeta.z_dim,
    );
    const plannerBuffer = await buffer(
      `models/${plannerModel}/planner.condition.${weightSuffix}.onnx`,
      "Loading FIBER scene encoder",
    );
    $("#load-note").textContent = "Preparing the planner…";
    const stepBuffer = await buffer(
      `models/${plannerModel}/planner.step.gpu.${weightSuffix}.onnx`,
      "Loading FIBER planner",
    );
    const timeBuffer = await buffer(
      `models/${plannerModel}/planner.time.${weightSuffix}.onnx`,
      "Loading cached timestep layers",
    );
    $("#load-note").textContent = "Preparing the planner…";
    planner = await createPlannerSession(
      plannerBuffer,
      stepBuffer,
      timeBuffer,
      options,
      backend === "webgpu",
      solverSteps,
    );
    $("#load-note").textContent = "Warming the planner and depth camera…";
    const c = meta.config;
    const warmValue = (data, dims, type = "float32") => ({
      type,
      dims,
      data: type === "bool" ? Uint8Array.from(data) : Float32Array.from(data),
    });
    await planner.sample({
      feed: {
        depth: warmValue(depth(), [1, 1, c.depth_height, c.depth_width]),
        state: warmValue(neutralState, [1, c.state_dim]),
        command: warmValue(new Float32Array(c.command_dim), [1, c.command_dim]),
        language: warmValue(meta.language.walk, [1, c.language_dim]),
        drop_command: warmValue([true], [1], "bool"),
        drop_state: warmValue([false], [1], "bool"),
        hand_state: warmValue(new Float32Array(c.hand_state_dim), [
          1,
          c.hand_state_dim,
        ]),
        hand_mask: warmValue([false], [1], "bool"),
      },
      trajectory: new Float32Array(c.num_knots * c.z_dim),
      handTrajectory: new Float32Array(c.num_knots * c.hand_dim),
      trajectoryDims: [1, c.num_knots, c.z_dim],
      handDims: [1, c.num_knots, c.hand_dim],
      steps: solverSteps[0],
    }); // Warm inference only: no physics step, sampler advance, or state dropout.
    ready = true;
    $("#save-run-report").disabled = false;
    pose();
    $("#loading").hidden = true;
    $("#pause").disabled = $("#reset").disabled = false;
    $("#runtime").textContent =
      `FIBER · ${useTeacher ? "teacher" : "student"} · ${backend.toUpperCase()}`;
    $("#runtime").title =
      `FP32 actor (${actorBackend.toUpperCase()}) · ${useF16Storage ? "FP16 weights / FP32 computation" : "W8A32"} planner (${backend.toUpperCase()}) · synchronous 1 Hz planning`;
    status(
      "STANDING",
      sceneName === "grasp"
        ? "Choose Grasp object to start the Dex3 interaction."
        : "Select the box or choose a task to begin.",
    );
    window.fiberDebug = {
      get state() {
        return {
          ready,
          sceneLoading,
          busy,
          running,
          sceneName,
          mode,
          phase,
          simTime,
          plannerMs,
          stateDropPlans,
          stateDropPending: shouldDropState(),
          lastExecutedCommand,
          goal: goal ? [...goal] : null,
          walkSelected,
          navigationIntent,
          holdingBox,
          doorHandContactStats: { ...doorHandContactStats },
          yawRate,
          activeBox,
          pickupLayout,
          pickupSpawn,
          goalRevision,
          goalYaw,
          goalHeight,
          filteredHeight,
          maxLift,
          heldFrames,
          currentContactForce,
          contactFrames: contactWindow.filter(Boolean).length,
          conditionedWaypoints,
          backend,
          actorBackend,
          build: "2026-09-29-student-1hz-threads-door-align",
          plannerModel,
          plannerThreads: planner?.wasmThreads ?? 1,
          crossOriginIsolated: window.crossOriginIsolated,
          planningHz: 50 / controlsPerPlan,
          rtc: { enabled: false },
          browserPreset: sceneConfig.browser_preset || "standard",
          locomotionOverride: useLocomotionBank(),
          locomotionBank: locomotionBank?.meta,
          solverSteps:
            stepOverride ??
            (mode === "grasp_table"
              ? sceneConfig.grasp_solver_steps
              : undefined) ??
            (useTeacher
              ? walkSelected && task !== "turn" && mode === "walk"
                ? 4
                : 8
              : 3),
          plannerCheckpoint: meta.source_checkpoint,
          plannerSHA256: meta.source_sha256,
          precision: {
            actor: "FP32",
            planner: useF16Storage
              ? "FP16 weights / FP32 computation"
              : "W8A32",
          },
          modelFiles: { ...loadedModelFiles },
          modelBytes: loadedModelBytes,
          sampler:
            (mode === "grasp_table" &&
              graspSwitchSamplerActive &&
              sceneConfig.switch_sampler_noise) ||
            sceneConfig.sampler_noise ||
            sceneName,
          userAgent: navigator.userAgent,
          error: $("#status").textContent,
        };
      },
      get rolloutSnapshot() {
        return {
          qpos: Array.from(data.qpos),
          qvel: Array.from(data.qvel),
          ctrl: Array.from(data.ctrl),
          warmstart: Array.from(data.qacc_warmstart),
          mocap_pos: Array.from(data.mocap_pos),
          mocap_quat: Array.from(data.mocap_quat),
          lastLatent: Array.from(lastLatent),
          lastAction: Array.from(lastAction),
          history: structuredClone(history),
        };
      },
      get lastPlanAudit() {
        return structuredClone(lastPlanAudit);
      },
      get events() {
        return structuredClone(runEvents);
      },
      async auditConfigure({ steps, noise, xy } = {}) {
        if (running || busy || sceneLoading)
          throw new Error("Pause before configuring an audit reset");
        if (steps !== undefined) {
          if (!Number.isInteger(steps) || steps < 2 || steps > 32)
            throw new Error("Invalid solver steps");
          stepOverride = steps;
        }
        if (noise) {
          if (!noiseCache.has(noise)) {
            const r = await fetch(`models/noise/${noise}.f32`);
            if (!r.ok) throw new Error("Missing audit sampler");
            noiseCache.set(noise, new Float32Array(await r.arrayBuffer()));
          }
          samplerNoise = noiseCache.get(noise);
        }
        if (xy) {
          sceneConfig.qpos[0] = xy[0];
          sceneConfig.qpos[1] = xy[1];
        }
        reset(false);
      },
      observations,
      depth,
      command,
      reset,
      choose,
      run,
      get objectScreenPoint() {
        const p = boxCenter().project(camera),
          r = canvas.getBoundingClientRect();
        return {
          x: r.left + ((p.x + 1) * r.width) / 2,
          y: r.top + ((1 - p.y) * r.height) / 2,
        };
      },
      get data() {
        return data;
      },
      get model() {
        return model;
      },
      get sceneConfig() {
        return sceneConfig;
      },
      get meta() {
        return meta;
      },
      get actor() {
        return actor;
      },
      get planner() {
        return planner;
      },
    };
  } catch (error) {
    console.error(error);
    $("#launch").disabled = false;
    $("#launch").textContent = "Retry launch";
    $("#load-note").textContent = error.message;
    status("LOAD FAILED", error.message);
  }
}
$("#launch").onclick = launch;
$("#pause").onclick = () => {
  if (running) {
    running = false;
    $("#pause").textContent = "Run";
  } else run();
};
$("#reset").onclick = () => reset();
$("#follow").onclick = () => {
  follow = !follow;
  $("#follow").setAttribute("aria-pressed", String(follow));
};
$("#fullscreen").onclick = () =>
  document.fullscreenElement
    ? document.exitFullscreen()
    : document.documentElement.requestFullscreen();
document
  .querySelectorAll("[data-task]")
  .forEach((b) => (b.onclick = () => choose(b.dataset.task)));
document.querySelectorAll("[data-scene]").forEach(
  (b) =>
    (b.onclick = async () => {
      if (sceneLoading) return;
      const resume = running;
      if (mj) sceneLoading = true;
      if (busy) {
        running = false;
        await sleep(100);
        while (busy) await sleep(100);
      }
      const requestedScene = b.dataset.scene;
      if (mj) {
        await scene(requestedScene);
        const o = observations();
        neutralState = Float32Array.from(
          [...o.q, ...o.v],
          (x, i) =>
            (x - meta.state_mean[i]) / Math.max(meta.state_std[i], 1e-6),
        );
        if (resume) run();
      } else {
        sceneName = requestedScene;
        document
          .querySelectorAll("[data-scene]")
          .forEach((btn) => btn.classList.toggle("active", btn === b));
      }
    }),
);
const ray = new THREE.Raycaster(),
  mouse = new THREE.Vector2();
let down,
  draggingGoal = null,
  draggingBox = null;
const throwGuide = new THREE.ArrowHelper(
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(),
  1,
  0xc6a9ff,
  0.16,
  0.08,
);
throwGuide.visible = false;
world.add(throwGuide);
function updateThrowDrag(e) {
  pointerRay(e);
  const point = ray.ray.intersectPlane(draggingBox.plane, new THREE.Vector3());
  if (!point) return;
  draggingBox.delta.copy(point).sub(draggingBox.start);
  const distance = draggingBox.delta.length();
  throwGuide.position.copy(boxCenter());
  throwGuide.visible = distance > 0.03;
  if (distance > 0.03) {
    const direction = draggingBox.delta
      .clone()
      .setZ(Math.min(0.8, distance * 0.5))
      .normalize();
    throwGuide.setDirection(direction);
    throwGuide.setLength(Math.min(2, distance), 0.16, 0.08);
  }
  needsRender = true;
}
function finishThrow(cancel = false) {
  if (!draggingBox) return;
  const delta = draggingBox.delta.clone();
  draggingBox = null;
  throwGuide.visible = false;
  down = null;
  canvas.style.cursor = "grab";
  if (cancel || delta.length() < 0.04) return;
  // User-applied impulse only: preserve position, mass, contact, and collisions.
  const joint = model.body_jntadr[sceneConfig.box_body],
    v = model.jnt_dofadr[joint];
  const speed = Math.min(5, delta.length() * 3),
    direction = delta.normalize();
  data.qvel[v] += direction.x * speed;
  data.qvel[v + 1] += direction.y * speed;
  data.qvel[v + 2] += Math.min(3, 1 + speed * 0.45);
  data.qvel[v + 3] += -direction.y * speed * 0.8;
  data.qvel[v + 4] += direction.x * speed * 0.8;
  heldFrames = 0;
  contactWindow = [];
  holdingBox = false;
  runEvents.push({
    type: "throw",
    simTime,
    velocity: Array.from(data.qvel.slice(v, v + 6)),
  });
  choose("pickup");
}
function pointerRay(e) {
  const r = canvas.getBoundingClientRect();
  mouse.set(
    ((e.clientX - r.left) / r.width) * 2 - 1,
    (-(e.clientY - r.top) / r.height) * 2 + 1,
  );
  ray.setFromCamera(mouse, camera);
}
canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || !ready || sceneLoading) return;
  down = [e.clientX, e.clientY];
  if (sceneName === "pickup" && pickupLayout === "throw" && !walkSelected) {
    pointerRay(e);
    const hit = ray.intersectObjects(
      geoms.filter((g) => g.mesh.userData.box).map((g) => g.mesh),
    )[0];
    if (hit) {
      const center = boxCenter();
      const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -center.z);
      const start = ray.ray.intersectPlane(plane, new THREE.Vector3());
      if (!start) return;
      draggingBox = { start, delta: new THREE.Vector3(), plane };
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = "crosshair";
    }
    return;
  }
  if (!walkSelected) return;
  pointerRay(e);
  const hit = ray.intersectObject(floor)[0];
  if (!hit) return;
  draggingGoal = [hit.point.x, hit.point.y];
  canvas.setPointerCapture(e.pointerId);
  manualGoal(
    draggingGoal,
    Math.atan2(hit.point.y - data.qpos[1], hit.point.x - data.qpos[0]),
  );
});
canvas.addEventListener("pointermove", (e) => {
  if (draggingBox) {
    updateThrowDrag(e);
    return;
  }
  if (!draggingGoal) return;
  pointerRay(e);
  const hit = ray.intersectObject(floor)[0];
  if (!hit) return;
  const dx = hit.point.x - draggingGoal[0],
    dy = hit.point.y - draggingGoal[1];
  if (Math.hypot(dx, dy) > 0.06) manualGoal(draggingGoal, Math.atan2(dy, dx));
});
canvas.addEventListener("pointerup", (e) => {
  if (draggingBox) {
    updateThrowDrag(e);
    finishThrow();
    return;
  }
  if (draggingGoal) {
    draggingGoal = null;
    down = null;
    return;
  }
  if (
    !ready ||
    e.button !== 0 ||
    !down ||
    Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6
  )
    return;
  down = null;
  pointerRay(e);
  const hit = ray.intersectObjects(geoms.map((g) => g.mesh))[0];
  if (hit?.object.userData.box) {
    selectObject(
      sceneName === "grasp"
        ? graspObjectChoice
        : sceneName === "pickup"
          ? pickupCaseChoice
          : hit.object.userData.boxIndex,
    );
    return;
  }
});
canvas.addEventListener("pointercancel", () => {
  finishThrow(true);
  draggingGoal = null;
  down = null;
});
window.addEventListener("keydown", (e) => {
  if (!ready || ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName))
    return;
  const key = e.key.toLowerCase();
  if (key === "r") {
    reset();
    return;
  }
  if (!"wasdqe".includes(key) || key.length !== 1) return;
  e.preventDefault();
  if (e.repeat) return;
  heldKeys.set(key, performance.now());
  const yaw = rootYaw();
  if (key === "q" || key === "e")
    manualGoal(
      goal || [data.qpos[0], data.qpos[1]],
      (goalYaw ?? yaw) + ((key === "q" ? 1 : -1) * Math.PI) / 6,
      !goal || task === "turn",
    );
  else {
    const a = yaw + { w: 0, s: Math.PI, a: Math.PI / 2, d: -Math.PI / 2 }[key];
    manualGoal([data.qpos[0] + Math.cos(a), data.qpos[1] + Math.sin(a)], yaw);
  }
});
window.addEventListener("keyup", (e) => heldKeys.delete(e.key.toLowerCase()));
window.addEventListener("blur", () => heldKeys.clear());
for (const id of [
  "waypoint-x",
  "waypoint-y",
  "waypoint-yaw",
  "waypoint-height",
])
  $("#" + id).addEventListener("input", () => {
    if (!goal) return;
    const x = Number($("#waypoint-x").value),
      y = Number($("#waypoint-y").value),
      yaw = (Number($("#waypoint-yaw").value) * Math.PI) / 180;
    const height = Number($("#waypoint-height").value);
    if ([x, y, yaw, height].every(Number.isFinite)) {
      goalHeight = Math.max(0.4, Math.min(1, height));
      manualGoal([x, y], yaw);
    }
  });
$("#yaw-rate").oninput = (e) => {
  yawRate = Number(e.target.value);
  $("#yaw-rate-value").textContent = `${yawRate.toFixed(1)} rad/s`;
  goalRevision++;
};
$("#pickup-layout").onchange = async (e) => {
  if (sceneLoading) return;
  if (mj) sceneLoading = true;
  const resume = running;
  running = false;
  requestVersion++;
  while (busy) await sleep(50);
  pickupLayout = e.target.value;
  document.querySelectorAll("[data-layout]").forEach((b) => {
    b.classList.toggle("active", b.dataset.layout === pickupLayout);
    b.setAttribute("aria-pressed", String(b.dataset.layout === pickupLayout));
  });
  if (mj) await scene("pickup");
  if (resume) run();
};

if (new URLSearchParams(location.search).has("autostart")) launch();

for (const button of document.querySelectorAll("[data-layout]"))
  button.onclick = () => {
    $("#pickup-layout").value = button.dataset.layout;
    $("#pickup-layout").dispatchEvent(new Event("change"));
  };

$("#save-run-report").onclick = () => {
  if (!ready) return;
  const report = {
    state: window.fiberDebug.state,
    events: runEvents,
    scene: {
      provenance: sceneConfig.provenance,
      object: sceneConfig.boxes?.[activeBox]?.label,
      xml: sceneConfig.xml,
    },
    qpos: Array.from(data.qpos),
    qvel: Array.from(data.qvel),
  };
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "fiber-run-report.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
