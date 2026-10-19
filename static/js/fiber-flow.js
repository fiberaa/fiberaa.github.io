/* FIBER animated 2D story. Each phase is its own composed scene; shared elements
   (Unitree G1 experts, the latent space, the planner, the frozen BFM, the
   executor) move between scenes so the method is told step by step:
   experts -> action-aligned projection -> collected latent fibers -> offline
   planner training -> planning -> execution. Robot clips are successful
   simulator rollouts; the depth and joint traces come from the same pickup
   rollout; diagrams are schematic, not recorded model output. Formulas are
   typeset with KaTeX in an overlay synced to the canvas. Reduced motion
   renders a single static frame. */
(() => {
  const root = document.querySelector(".fiber-overview");
  if (!root) return;
  const stage = root.querySelector("[data-fiber-flow]");
  if (!stage) return;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");

  const phaseEl = root.querySelector("[data-fiber-phase]");
  const detailEl = root.querySelector("[data-fiber-detail]");
  const playBtn = root.querySelector("[data-fiber-play]");
  const replayBtn = root.querySelector("[data-fiber-replay]");
  const slider = root.querySelector("[data-fiber-time]");
  const jumps = [...root.querySelectorAll("[data-fiber-jump]")];

  const titles = [
    "Heterogeneous skill acquisition",
    "Action-aligned latent policy projection",
    "Closed-loop latent rollouts",
    "Offline planner training",
    "Unified latent flow planning",
    "Execution through the frozen BFM",
  ];
  const details = [
    "Interaction experts are acquired independently with task-appropriate supervision: motion imitation for box pickup and object grasping, staged reward shaping for door opening.",
    "Latent Policy Projection learns a projector $P_\\psi(o_{t_k}) \\to z_k$ whose induced BFM actions $\\pi_{\\mathrm{BFM}}(s, z)$ match the expert actions $a^E$ over a short horizon $H_P$; the semantic term $\\lVert z_k - B(s_{t_k}) \\rVert^2$ anchors $z_k$ to the BFM's native behavior and resolves action-equivalent latents.",
    "Rolling out each projector through the frozen BFM collects closed-loop latent trajectories $\\mathcal{D}^Z_i = \\{(d_t, p_t, u_t, z_t)\\}$, fibers in the shared latent space. The original experts are no longer needed.",
    "One DiT planner is trained entirely offline with flow matching. Its input tokens are the noisy latent plan $Z_\\tau$ with flow time $\\tau$, a state token $p_t$, vision tokens from depth $d_t$, steering tokens $c = E_{\\mathrm{steer}}(u)$ and registers; it learns the velocity $v_\\theta$ that carries noise $\\varepsilon$ onto the latent trajectory $Z$.",
    "At deployment the planner reads onboard depth, proprioception and a steering token (a language command or a waypoint) and integrates the learned flow from noise to a future latent plan $\\hat{Z}_t = (z_t, \\dots, z_{t+H-1})$.",
    "The frozen BFM turns each planned latent and the live state $s_t$ into whole-body actions at 50 Hz with high-rate state feedback, while the planner replans at 5 Hz from new observations.",
  ];
  const DUR = [4.6, 7.6, 5.0, 5.6, 5.4, 5.6]; // seconds per scene
  const START = DUR.reduce((a, d, i) => (a.push(i ? a[i - 1] + DUR[i - 1] : 0), a), []);
  const TOTAL = DUR.reduce((a, b) => a + b, 0);

  const C = {
    pickup: "#c07d54", door: "#4f7fb0", grasp: "#3f9d92",
    ink: "#2f4459", muted: "#7f8d97", faint: "#c5ced4",
    panel: "#ffffff", panelEdge: "#d3dbe0", latentFill: "#edf3f2", line: "#c9d2d8",
  };
  const expertCols = [C.pickup, C.door, C.grasp];
  const expertKeys = ["pickup", "door", "grasp"];
  const expertNames = ["Box pickup", "Door opening", "Object grasping"];

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  stage.appendChild(canvas);
  const mathLayer = document.createElement("div");
  mathLayer.className = "fiber-math-layer";
  mathLayer.setAttribute("aria-hidden", "true");
  stage.appendChild(mathLayer);

  const VW = 1100, VH = 470;
  let cw = 0, ch = 0, dpr = 1, vs = 1, vox = 0, voy = 0; // virtual -> CSS px
  function resize() {
    cw = stage.clientWidth || 640;
    ch = stage.clientHeight || Math.round((cw * VH) / VW);
    dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
    vs = Math.min(cw / VW, ch / VH);
    vox = (cw - VW * vs) / 2; voy = (ch - VH * vs) / 2;
  }

  // ---- assets (lazy: start loading when the figure approaches) ----------
  const anim = {};
  const inputs = { depth: new Image(), meta: null };
  let assetsRequested = false;
  function requestAssets() {
    if (assetsRequested) return; assetsRequested = true;
    ["pickup", "grasp", "door"].forEach((k) => {
      const a = (anim[k] = { img: new Image(), meta: null });
      a.img.decoding = "async";
      a.img.src = `static/images/fiber-atlas/g1-anim-${k}.webp`;
      // decode off the main thread before first use (avoids a hitch mid-animation)
      a.img.decode().then(() => (a.ready = true), () => (a.ready = a.img.complete));
      fetch(`static/images/fiber-atlas/g1-anim-${k}.json`).then((r) => r.json()).then((m) => (a.meta = m)).catch(() => {});
    });
    inputs.depth.src = "static/images/fiber-atlas/depth-pickup.webp";
    fetch("static/images/fiber-atlas/planner-inputs.json").then((r) => r.json()).then((m) => (inputs.meta = m)).catch(() => {});
  }

  // ---- helpers ---------------------------------------------------------
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp01 = (v) => Math.max(0, Math.min(1, v));
  const ease = (v) => (v = clamp01(v)) * v * (3 - 2 * v);
  const easeOut = (v) => 1 - Math.pow(1 - clamp01(v), 3);
  const rand = (i) => { const x = Math.sin(i * 127.1 + 3.7) * 43758.5; return x - Math.floor(x); };
  function bez(p0, c0, c1, p1, t) {
    const u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return { x: a * p0.x + b * c0.x + c * c1.x + d * p1.x, y: a * p0.y + b * c0.y + c * c1.y + d * p1.y };
  }
  function roundRect(x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function dot(x, y, r, color, alpha, halo = true) {
    if (alpha <= 0.01) return;
    ctx.fillStyle = color;
    if (halo) { ctx.globalAlpha = alpha * 0.22; ctx.beginPath(); ctx.arc(x, y, r * 2.4, 0, 7); ctx.fill(); }
    ctx.globalAlpha = alpha; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
    ctx.globalAlpha = 1;
  }
  function ring(x, y, r, color, alpha, w = 2) {
    if (alpha <= 0.01) return;
    ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = w;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke(); ctx.restore();
  }
  function curve(p0, c0, c1, p1, color, alpha, w = 1.2, dash) {
    if (alpha <= 0.01) return;
    ctx.save(); ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = w;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.bezierCurveTo(c0.x, c0.y, c1.x, c1.y, p1.x, p1.y); ctx.stroke();
    ctx.restore();
  }
  function arrowHead(x, y, ang, color, alpha, s = 7) {
    if (alpha <= 0.01) return;
    ctx.save(); ctx.globalAlpha = alpha; ctx.fillStyle = color; ctx.translate(x, y); ctx.rotate(ang);
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-s, -s * 0.55); ctx.lineTo(-s, s * 0.55); ctx.closePath(); ctx.fill();
    ctx.restore();
  }
  function stream(p0, c0, c1, p1, color, alpha, n, r, off = 0, speed = 0.5) {
    if (alpha <= 0.02) return;
    for (let i = 0; i < n; i++) {
      const t = ((flow * speed + i / n + off) % 1 + 1) % 1;
      const p = bez(p0, c0, c1, p1, t);
      dot(p.x, p.y, r, color, alpha * Math.sin(t * Math.PI));
    }
  }
  const FONT = (size, weight, italic) => `${italic ? "italic " : ""}${weight} ${size}px Inter, system-ui, sans-serif`;
  function label(text, x, y, size, color, align = "center", weight = 500, alpha = 1, italic = false, halo = false) {
    if (alpha * TXT <= 0.01) return;
    ctx.save();
    ctx.globalAlpha = clamp01(alpha) * TXT;
    ctx.font = FONT(size, weight, italic);
    ctx.textAlign = align; ctx.textBaseline = "middle";
    if (halo) { ctx.lineJoin = "round"; ctx.lineWidth = 5; ctx.strokeStyle = "rgba(234,239,242,0.95)"; ctx.strokeText(text, x, y); }
    ctx.fillStyle = color; ctx.fillText(text, x, y);
    ctx.restore();
  }
  const textW = (text, size, weight = 500) => { ctx.save(); ctx.font = FONT(size, weight); const w = ctx.measureText(text).width; ctx.restore(); return w; };

  // ---- KaTeX overlay: formulas typeset in HTML, positioned per frame ------
  const mathPool = new Map(); let mathUsed = new Set();
  function mathEl(id, tex) {
    let e = mathPool.get(id);
    if (!e) { e = { el: document.createElement("span"), tex: null, w: 0, h: 0 }; e.el.className = "fiber-math"; mathLayer.appendChild(e.el); mathPool.set(id, e); }
    if (e.tex !== tex || (!e.typeset && window.katex)) {
      e.tex = tex; e.typeset = !!window.katex;
      if (e.typeset) window.katex.render(tex, e.el, { throwOnError: false, displayMode: false });
      else e.el.textContent = "";
      e.w = 0;
    }
    return e;
  }
  // width of a formula in virtual px at a given size (measured once, cached)
  function mathW(id, tex, size) {
    const e = mathEl(id, tex);
    if (!e.w) { e.el.style.fontSize = "100px"; e.el.style.opacity = "0"; e.el.style.display = "block"; e.w = e.el.getBoundingClientRect().width / 100; }
    return e.w * size;
  }
  function math(id, tex, x, y, size, color, align = "center", alpha = 1) {
    const a = clamp01(alpha) * TXT;
    const e = mathEl(id, tex);
    mathUsed.add(id);
    if (a <= 0.01) { e.el.style.opacity = "0"; return; }
    const w = mathW(id, tex, size);
    const ax = align === "left" ? 0 : align === "right" ? w : w / 2;
    const st = e.el.style;
    st.display = "block"; st.opacity = String(a); st.color = color;
    // typeset at the virtual size and scale as a whole: tiny font sizes would
    // otherwise pixel-snap KaTeX glyph spacing on small screens
    st.fontSize = `${size}px`;
    st.transform = `translate(${(vox + (x - ax) * vs).toFixed(1)}px, ${(voy + y * vs).toFixed(1)}px) translateY(-50%) scale(${vs.toFixed(4)})`;
  }
  // "text  formula" as one centered/aligned unit (Inter words + KaTeX math)
  function labelMath(id, text, tex, x, y, size, color, align = "center", weight = 600, alpha = 1, gap = 5) {
    const tw = textW(text, size, weight), mw = mathW(id, tex, size * 1.08), W = tw + gap + mw;
    const x0 = align === "left" ? x : align === "right" ? x - W : x - W / 2;
    label(text, x0, y, size, color, "left", weight, alpha);
    math(id, tex, x0 + tw + gap, y, size * 1.08, color, "left", alpha);
  }
  function mathFrameEnd() {
    for (const [id, e] of mathPool) if (!mathUsed.has(id) && e.el.style.opacity !== "0") e.el.style.opacity = "0";
    mathUsed = new Set();
  }

  function box(x, y, w, h, alpha, title, subTex, accent, lock = false, id = "box", subText = null) {
    if (alpha <= 0.01) return;
    ctx.save();
    ctx.globalAlpha = alpha * 0.08; ctx.fillStyle = "#2a4052";
    roundRect(x + 1, y + 4, w, h, 12); ctx.fill();                 // cheap soft shadow
    ctx.globalAlpha = alpha;
    roundRect(x, y, w, h, 12); ctx.fillStyle = C.panel; ctx.fill();
    ctx.lineWidth = 1.4; ctx.strokeStyle = accent || C.panelEdge; ctx.stroke();
    if (lock) {
      const lx = x + w - 17, ly = y + 11;
      ctx.strokeStyle = C.muted; ctx.lineWidth = 1.3; ctx.fillStyle = C.muted;
      ctx.beginPath(); ctx.arc(lx + 4, ly + 4, 3, Math.PI, 0); ctx.stroke(); ctx.fillRect(lx, ly + 4, 8, 6);
    }
    ctx.restore();
    if (title) label(title, x + w / 2, y + h / 2 + (subTex || subText ? -9 : 0), 14, C.ink, "center", 650, alpha);
    if (subTex) math(id + "-sub", subTex, x + w / 2, y + h / 2 + 12, 12.5, C.muted, "center", alpha);
    if (subText) label(subText, x + w / 2, y + h / 2 + 12, 11, C.muted, "center", 500, alpha);
  }

  // Robot clips play forward once per scene and hold on the successful end
  // pose. Restarts are scheduled only at scene cuts (and paused-scene loops):
  // an invisible robot resets instantly, a visible one dips out and back in.
  const clips = {};
  function clipRestart(id, at, rate = 1) { (clips[id] || (clips[id] = { start: -1e9, rate: 1, dip: null })).pending = { at, rate }; }
  function clipFrame(m, start, rate, T) {
    const play = m.sim_seconds / (m.speed || 1.15), v = ((T - start) * rate) / play;
    return v <= 0 ? 0 : v >= 1 ? m.frames - 1 : v * (m.frames - 1);
  }
  // h = ROBOT height; cx = robot center; footY = ground.
  function drawG1(id, key, cx, footY, h, alpha, clipTop = 0) {
    const c = clips[id] || (clips[id] = { start: -1e9, rate: 1, dip: null });
    if (c.pending && animClock >= c.pending.at) {
      const q = c.pending; c.pending = null;
      if (alpha < 0.05) { c.start = q.at; c.dip = null; }
      else { c.dip = { start: c.start, rate: c.rate, t: animClock }; c.start = animClock + 0.5; }
      c.rate = q.rate;
    }
    if (alpha <= 0.01) return;
    const a = anim[key], m = a && a.meta;
    if (!m || !a.ready || !a.img.naturalWidth) return;
    const N = m.frames;
    const frameH = h / m.rh, frameW = frameH * (m.fw / m.fh);
    const left = cx - m.rcx * frameW, top = footY - m.rfoot * frameH;
    const put = (i, al) => {
      if (al <= 0.01) return;
      ctx.globalAlpha = alpha * al;
      ctx.drawImage(a.img, (i % m.cols) * m.fw, Math.floor(i / m.cols) * m.fh, m.fw, m.fh, left, top, frameW, frameH);
    };
    ctx.save();
    const sw = h * 0.3, g = ctx.createRadialGradient(cx, footY, 0, cx, footY, sw);
    g.addColorStop(0, "rgba(40,55,70,0.9)"); g.addColorStop(1, "rgba(40,55,70,0)");
    ctx.globalAlpha = alpha * 0.14; ctx.fillStyle = g;
    ctx.beginPath(); ctx.ellipse(cx, footY, sw, h * 0.04, 0, 0, 7); ctx.fill();
    ctx.restore();
    if (clipTop) { ctx.save(); ctx.beginPath(); ctx.rect(left - 4, footY - h * clipTop, frameW + 8, h * clipTop + frameH); ctx.clip(); }
    const putF = (f, al) => { const i0 = Math.floor(f), fr = f - i0; put(i0, al); if (fr > 0.02 && i0 + 1 < N) put(i0 + 1, al * fr); };
    if (reduced.matches) put(N - 1, 1);
    else if (c.dip && animClock < c.dip.t + 0.5) {
      const v = (animClock - c.dip.t) / 0.5;
      if (v < 0.5) putF(clipFrame(m, c.dip.start, c.dip.rate, animClock), 1 - ease(v * 2)); else put(0, ease(v * 2 - 1));
    } else putF(clipFrame(m, c.start, c.rate, animClock), 1);
    if (clipTop) ctx.restore();
    ctx.globalAlpha = 1;
  }
  const fiberDefs = [{ s: 0.3, c: C.pickup, yo: -0.36 }, { s: 2.1, c: C.door, yo: 0.02 }, { s: 4.0, c: C.grasp, yo: 0.38 }];
  const fiberPath = (f, x, y, w, h) => (t) => ({ x: x + 26 + t * (w - 52), y: y + h / 2 + 18 + f.yo * h * 0.3 + Math.sin(t * Math.PI * 2.2 + f.s) * h * 0.11 + Math.sin(t * 5 + f.s) * h * 0.03 });
  // grow[i] in [0,1]: how much of fiber i has been traced out (rollout progress)
  function fibers(x, y, w, h, alpha, density, hl = -1, grow = null) {
    if (alpha <= 0.02) return;
    fiberDefs.forEach((f, i) => {
      const path = fiberPath(f, x, y, w, h), dim = hl >= 0 && i !== hl ? 0.4 : 1, g = grow ? grow[i] : 1;
      if (g <= 0.001) return;
      ctx.save(); ctx.strokeStyle = f.c; ctx.globalAlpha = 0.26 * alpha * dim * (i === hl ? 2 : 1); ctx.lineWidth = i === hl ? 2.2 : 1.5;
      ctx.beginPath(); for (let t = 0; t <= g + 1e-4; t += 0.01) { const p = path(Math.min(t, g)); t === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y); } ctx.stroke(); ctx.restore();
      const n = Math.round(15 * density);
      for (let k = 0; k < n; k++) {
        const t = ((flow * 0.42 + k / n + i * 0.2) % 1 + 1) % 1; if (t > g) continue;
        const p = path(t); dot(p.x, p.y, 2.6, f.c, alpha * dim * (0.45 + 0.55 * Math.sin(t * Math.PI)) * clamp01((g - t) / 0.04));
      }
      if (g < 0.999) { const p = path(g); dot(p.x, p.y, 4.2, f.c, alpha); }   // the rollout's current latent
    });
  }

  // ---- scene layout targets --------------------------------------------
  function targets(p) {
    const t = {
      ex: [{ x: 118, fy: 150, h: 118, a: 0 }, { x: 118, fy: 285, h: 118, a: 0 }, { x: 118, fy: 420, h: 118, a: 0 }],
      exe: { x: 905, fy: 408, h: 250, a: 0 },
      lat: { x: 300, y: 60, w: 470, h: 350, a: 0 },
      pl: { x: 650, y: 187, w: 200, h: 96, a: 0 },
      bfm: { x: 450, y: 205, w: 160, h: 62, a: 0 },
      ov: { lpp: 0, collect: 0, train: 0, inputs: 0, pred: 0, plan: 0, exec: 0, feedback: 0 },
    };
    if (p === 0) {
      t.ex = [{ x: 262, fy: 402, h: 212, a: 1 }, { x: 540, fy: 402, h: 212, a: 1 }, { x: 858, fy: 402, h: 212, a: 1 }];
    } else if (p === 1) {
      t.ex[0] = { x: 128, fy: 402, h: 232, a: 1 };
      t.ov.lpp = 1;
    } else if (p === 2) {
      t.ex = [{ x: 118, fy: 150, h: 118, a: 1 }, { x: 118, fy: 285, h: 118, a: 1 }, { x: 118, fy: 420, h: 118, a: 1 }];
      t.lat = { x: 360, y: 58, w: 610, h: 350, a: 1 };
      t.ov.collect = 1;
    } else if (p === 3) {
      t.lat = { x: 30, y: 92, w: 262, h: 250, a: 1 };
      t.pl = { x: 690, y: 124, w: 270, h: 92, a: 1 };
      t.ov.train = 1;
    } else if (p === 4) {
      t.pl = { x: 420, y: 187, w: 200, h: 96, a: 1 };
      t.ov.inputs = 1; t.ov.pred = 1;
    } else {
      t.pl = { x: 110, y: 186, w: 190, h: 96, a: 1 };
      t.bfm = { x: 430, y: 203, w: 170, h: 62, a: 1 };
      t.exe = { x: 880, fy: 405, h: 250, a: 1 };
      t.ov.plan = 1; t.ov.exec = 1; t.ov.feedback = 1;
    }
    return t;
  }
  // Staged transitions. Alpha: outgoing fades fast in place; incoming waits
  // for a gate, then fades in. Geometry: critically damped springs (smooth
  // start and stop); an invisible element teleports to its new slot.
  const K = (dt, tau) => 1 - Math.exp(-dt / tau);
  function stepAlpha(cur, tgt, dt, gate) {
    if (tgt < cur) return cur + (tgt - cur) * K(dt, 110);
    const g = tgt * gate; return g > cur ? cur + (g - cur) * K(dt, 190) : cur;
  }
  const W0 = 8.5; // spring stiffness (rad/s): ~0.55 s settle
  function smoothEl(cur, vel, tg, dt, gate) {
    cur.a = stepAlpha(cur.a, tg.a, dt, gate);
    if (tg.a <= 0.03) { for (const k in vel) vel[k] = 0; return; }
    const snap = cur.a < 0.04, h = dt / 1000;
    for (const key in tg) {
      if (key === "a") continue;
      if (snap) { cur[key] = tg[key]; vel[key] = 0; continue; }
      const v = vel[key] || 0, acc = W0 * W0 * (tg[key] - cur[key]) - 2 * W0 * v;
      vel[key] = v + acc * h; cur[key] += vel[key] * h;
    }
  }
  function smoothAll(cur, vel, tg, dt, gate) {
    tg.ex.forEach((e, i) => smoothEl(cur.ex[i], vel.ex[i], e, dt, gate));
    ["exe", "lat", "pl", "bfm"].forEach((n) => smoothEl(cur[n], vel[n], tg[n], dt, gate));
    for (const key in tg.ov) cur.ov[key] = stepAlpha(cur.ov[key], tg.ov[key], dt, gate);
  }
  const S = JSON.parse(JSON.stringify(targets(0)));
  const V = { ex: [{}, {}, {}], exe: {}, lat: {}, pl: {}, bfm: {} };

  // ---- scene 1: action-aligned latent policy projection ----------------
  function drawLPP(a, u) {
    if (a <= 0.02) return;
    const conv = easeOut(clamp01((u - 0.1) / 0.62));
    const dyn = playing || reduced.matches ? a : a * clamp01(Math.min(u / 0.06, (1 - u) / 0.06));
    const L = { x: 420, y: 250, r: 118 };
    const zs = { x: L.x + 18, y: L.y + 22 }, alt = { x: L.x - 58, y: L.y - 52 }, bs = { x: zs.x + 20, y: zs.y - 16 };
    ctx.save(); ctx.globalAlpha = a;
    const g = ctx.createRadialGradient(L.x, L.y, 10, L.x, L.y, L.r);
    g.addColorStop(0, "#f4f8f7"); g.addColorStop(1, "#e3ecea");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(L.x, L.y, L.r, 0, 7); ctx.fill();
    ctx.lineWidth = 1.3; ctx.strokeStyle = C.panelEdge; ctx.stroke();
    ctx.beginPath(); ctx.arc(L.x, L.y, L.r - 1, 0, 7); ctx.clip();
    [[zs, 1], [alt, 0.55]].forEach(([c, w]) => {
      for (let k = 1; k <= 5; k++) {
        ctx.strokeStyle = C.grasp; ctx.globalAlpha = a * 0.13 * w * (1.2 - k * 0.15); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.ellipse(c.x, c.y, 14 * k, 10 * k, -0.5, 0, 7); ctx.stroke();
      }
    });
    ctx.restore();
    ctx.save(); ctx.globalAlpha = a * 0.8; ctx.strokeStyle = C.door; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.arc(bs.x, bs.y, 20, 0, 7); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.door; ctx.translate(bs.x, bs.y); ctx.rotate(Math.PI / 4); ctx.fillRect(-3.5, -3.5, 7, 7); ctx.restore();
    math("lpp-bs", "B(s_{t_k})", bs.x + 28, bs.y - 16, 13, C.door, "left", a);
    label("action-equivalent", alt.x, alt.y - 36, 10, C.muted, "center", 500, a * 0.9);
    const zAt = (v) => { const R = 92 * Math.pow(1 - v, 1.35), th = 2.55 - 2.3 * v; return { x: zs.x + R * Math.cos(th), y: zs.y + R * Math.sin(th) * 0.8 }; };
    for (let k = 0; k < 24; k++) { const p = zAt(conv * (k / 24)); dot(p.x, p.y, 1.5, C.grasp, dyn * 0.4 * (k / 24), false); }
    const z = zAt(conv);
    // semantic anchor: L_sem pulls z_k toward B(s_tk)
    ctx.save(); ctx.globalAlpha = dyn * 0.6; ctx.strokeStyle = C.door; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(z.x, z.y); ctx.lineTo(bs.x, bs.y); ctx.stroke(); ctx.restore();
    math("lpp-lsem", "\\mathcal{L}_{\\mathrm{sem}} = \\lVert z_k - B(s_{t_k}) \\rVert^2", L.x - 6, L.y + 84, 11, C.door, "center", a * 0.95);
    dot(z.x, z.y, 5.2, C.grasp, dyn);
    math("lpp-z", "z_k", z.x - 13, z.y - 13, 15, C.grasp, "center", dyn);
    label("BFM latent space", L.x, L.y - L.r - 16, 13.5, C.ink, "center", 650, a);
    labelMath("lpp-proj", "Projector", "P_\\psi(o_{t_k}) \\rightarrow z_k", L.x, L.y + L.r + 19, 11.5, C.muted, "center", 500, a);

    curve({ x: 205, y: 262 }, { x: 240, y: 262 }, { x: 260, y: 258 }, { x: L.x - L.r - 6, y: 256 }, C.line, a, 1.3);
    arrowHead(L.x - L.r - 4, 256, 0, C.faint, a);
    labelMath("lpp-obs", "observation", "o_{t_k}", 250, 243, 10.5, C.muted, "center", 500, a, 4);

    const B = { x: 588, y: 222, w: 132, h: 58 };
    stream(z, { x: z.x + 60, y: z.y }, { x: B.x - 40, y: B.y + 29 }, { x: B.x, y: B.y + 29 }, C.grasp, dyn, 5, 2.4, 0, 0.7);
    box(B.x, B.y, B.w, B.h, a, "Frozen BFM", "\\pi_{\\mathrm{BFM}}(s, z)", null, true, "lpp-bfm");
    const A = { x: 760, y: 92, w: 300, h: 214 };
    stream({ x: B.x + B.w, y: B.y + 29 }, { x: B.x + B.w + 20, y: B.y + 29 }, { x: A.x - 20, y: 220 }, { x: A.x, y: 220 }, C.grasp, dyn, 3, 2.4, 0, 0.7);

    ctx.save(); ctx.globalAlpha = a;
    roundRect(A.x, A.y, A.w, A.h, 14); ctx.fillStyle = "rgba(255,255,255,0.82)"; ctx.fill();
    ctx.strokeStyle = conv > 0.97 ? C.grasp : C.panelEdge; ctx.lineWidth = 1.3; ctx.stroke();
    ctx.restore();
    label("Action space", A.x + 16, A.y + 18, 13, C.ink, "left", 650, a);
    math("lpp-lact", "\\mathcal{L}_{\\mathrm{act}} = \\frac{1}{H_P}\\sum_{h} \\big\\lVert \\pi_{\\mathrm{BFM}}(s_{t_k+h}, z_{t_k+h}) - a^{E}_{t_k+h} \\big\\rVert^2", A.x + A.w / 2, A.y - 22, 12, C.ink, "center", a);
    const H = 5, px = (k) => A.x + 44 + k * 54, ey = (k) => A.y + 118 - Math.sin(k * 0.9 + 0.4) * 34 - k * 4;
    const off = [[-18, 42], [16, 50], [-24, -38], [22, 36], [-14, -44]];
    ctx.save(); ctx.globalAlpha = a * 0.55; ctx.strokeStyle = C.pickup; ctx.lineWidth = 1.6;
    ctx.beginPath(); for (let k = 0; k < H; k++) { k ? ctx.lineTo(px(k), ey(k)) : ctx.moveTo(px(k), ey(k)); } ctx.stroke(); ctx.restore();
    const bp = []; for (let k = 0; k < H; k++) { const e = 1 - conv; bp.push({ x: px(k) + off[k][0] * e, y: ey(k) + off[k][1] * e }); }
    ctx.save(); ctx.globalAlpha = dyn * 0.5 * (0.35 + 0.65 * conv); ctx.strokeStyle = C.grasp; ctx.lineWidth = 1.4;
    ctx.beginPath(); bp.forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.stroke(); ctx.restore();
    const ticks = ["t_k", "t_k{+}1", "t_k{+}2", "\\cdots", "t_k{+}H_P{-}1"];
    for (let k = 0; k < H; k++) {
      ctx.save(); ctx.globalAlpha = dyn * 0.6 * (1 - conv); ctx.strokeStyle = C.muted; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(px(k), ey(k)); ctx.lineTo(bp[k].x, bp[k].y); ctx.stroke(); ctx.restore();
      ring(px(k), ey(k), 6.5, C.pickup, a);
      dot(bp[k].x, bp[k].y, 3.6, C.grasp, dyn, true);
      math(`lpp-t${k}`, ticks[k], px(k), A.y + A.h - 16, 11, C.muted, "center", a);
    }
    ring(A.x + A.w - 130, A.y + 18, 5, C.pickup, a);
    labelMath("lpp-lgE", "expert", "a^{E}", A.x + A.w - 120, A.y + 18, 11, C.pickup, "left", 600, a, 4);
    dot(A.x + A.w - 130, A.y + 38, 3.6, C.grasp, a, false);
    labelMath("lpp-lgB", "BFM", "\\pi_{\\mathrm{BFM}}", A.x + A.w - 120, A.y + 38, 11, C.grasp, "left", 600, a, 4);

    curve({ x: 170, y: 150 }, { x: 330, y: 36 }, { x: 640, y: 40 }, { x: A.x - 6, y: A.y + 70 }, C.pickup, a * 0.75, 1.4, [5, 4]);
    arrowHead(A.x - 5, A.y + 70, 0.6, C.pickup, a * 0.85);
    labelMath("lpp-exp", "expert actions", "a^{E}_t = \\pi^{E}_i(o_{i,t})", 470, 45, 11.5, C.pickup, "center", 600, a);

    const P = { x: A.x, y: 322, w: A.w, h: 120 };
    ctx.save(); ctx.globalAlpha = a;
    roundRect(P.x, P.y, P.w, P.h, 14); ctx.fillStyle = "rgba(255,255,255,0.82)"; ctx.fill(); ctx.strokeStyle = C.panelEdge; ctx.lineWidth = 1.3; ctx.stroke();
    ctx.globalAlpha = a * 0.7; ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(P.x + 34, P.y + 44); ctx.lineTo(P.x + 34, P.y + P.h - 24); ctx.lineTo(P.x + P.w - 14, P.y + P.h - 24); ctx.stroke(); ctx.restore();
    const lossAt = (v) => Math.exp(-4.2 * v) * 0.9 + 0.05 + 0.02 * Math.sin(v * 31) * (1 - v);
    const lx = (v) => P.x + 34 + v * (P.w - 52), ly = (v) => P.y + P.h - 24 - lossAt(v) * (P.h - 72);
    const prog = clamp01((u - 0.1) / 0.62);
    ctx.save(); ctx.globalAlpha = dyn; ctx.strokeStyle = C.grasp; ctx.lineWidth = 1.8;
    ctx.beginPath(); for (let v = 0; v <= prog + 1e-6; v += 0.01) { v ? ctx.lineTo(lx(v), ly(v)) : ctx.moveTo(lx(v), ly(v)); } ctx.stroke(); ctx.restore();
    dot(lx(prog), ly(prog), 3.4, C.grasp, dyn);
    math("lpp-loss", "\\mathcal{L}_{\\mathrm{proj}} = \\mathcal{L}_{\\mathrm{act}} + \\lambda_{\\mathrm{sem}}\\,\\mathcal{L}_{\\mathrm{sem}}", P.x + P.w / 2, P.y + 22, 12.5, C.ink, "center", a);
    label("training steps", P.x + P.w - 14, P.y + P.h - 11, 9.5, C.muted, "right", 500, a);
    labelMath("lpp-rs", "projector rollout states", "s_{t_k}", 128, 446, 11, C.muted, "center", 500, a, 4);
    const ma = a * ease((u - 0.72) / 0.12) * (conv > 0.97 ? 1 : 0) * (playing ? 1 : dyn / Math.max(a, 1e-3));
    if (ma > 0.01) {
      const bw = 74, bh = 19, bx = A.x + A.w - bw - 12, by = A.y + A.h - 48;
      ctx.save(); ctx.globalAlpha = ma * TXT; roundRect(bx, by, bw, bh, 9.5); ctx.fillStyle = "rgba(63,157,146,0.12)"; ctx.fill(); ctx.restore();
      label("✓ matched", bx + bw / 2, by + bh / 2 + 0.5, 10.5, C.grasp, "center", 650, ma);
    }
  }

  // ---- scene 3: flow matching from noise to a collected latent fiber ----
  function drawTrain(a, u, pl) {
    if (a <= 0.02) return;
    const cyc = (u * 2) % 1, mp = ease(clamp01((cyc - 0.14) / 0.58));
    const f = fiberDefs[2], lat = S.lat;
    const src = fiberPath(f, lat.x, lat.y, lat.w, lat.h);
    const x0 = 334, x1 = 560, yc = 205;
    // target Z = a window of the highlighted collected fiber, re-drawn at the training site
    const tgt = (t) => { const p = src(t); return { x: lerp(x0, x1, t), y: yc + (p.y - (lat.y + lat.h / 2 + 18 + f.yo * lat.h * 0.3)) * 1.2 }; };
    ctx.save(); ctx.globalAlpha = a * 0.3; ctx.strokeStyle = C.grasp; ctx.lineWidth = 1.4; ctx.setLineDash([3, 4]);
    ctx.beginPath(); for (let t = 0; t <= 1.0001; t += 0.02) { const p = tgt(t); t === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y); } ctx.stroke(); ctx.restore();
    const fadeCyc = cyc > 0.9 ? (1 - cyc) / 0.1 : cyc < 0.06 ? cyc / 0.06 : 1;
    // Z_tau = (1 - tau) eps + tau Z; each sample shows its target velocity Z - eps
    for (let i = 0; i < 48; i++) {
      const t = i / 47, p = tgt(t), dx = (rand(i) - 0.5) * 110, dy = (rand(i + 99) - 0.5) * 104;
      const nx = p.x + dx * (1 - mp), ny = p.y + dy * (1 - mp), vl = 0.16 * (1 - mp);
      if (vl > 0.01) {
        ctx.save(); ctx.globalAlpha = a * 0.35 * fadeCyc; ctx.strokeStyle = C.grasp; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(nx, ny); ctx.lineTo(nx - dx * vl, ny - dy * vl); ctx.stroke(); ctx.restore();
      }
      dot(nx, ny, 2.3, C.grasp, a * (0.3 + 0.6 * mp) * fadeCyc);
    }
    curve({ x: lat.x + lat.w - 12, y: lat.y + lat.h * 0.72 }, { x: lat.x + lat.w + 20, y: lat.y + lat.h * 0.8 }, { x: x0 - 34, y: yc + 60 }, { x: x0 - 4, y: yc + 40 }, C.grasp, a * 0.4, 1.1, [2, 4]);
    label("offline flow matching", (x0 + x1) / 2, 70, 12.5, C.ink, "center", 650, a);
    labelMath("tr-z", "target", "Z = (z_t, \\dots, z_{t+H-1})", (x0 + x1) / 2, 92, 11.5, C.grasp, "center", 600, a, 5);
    const ty = 332, tx0 = x0 + 22, tx1 = x1 - 22;
    ctx.save(); ctx.globalAlpha = a * 0.6; ctx.strokeStyle = C.faint; ctx.lineWidth = 2; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(tx0, ty); ctx.lineTo(tx1, ty); ctx.stroke();
    ctx.globalAlpha = a; ctx.strokeStyle = C.grasp; ctx.beginPath(); ctx.moveTo(tx0, ty); ctx.lineTo(lerp(tx0, tx1, mp), ty); ctx.stroke(); ctx.restore();
    dot(lerp(tx0, tx1, mp), ty, 3.6, C.grasp, a * fadeCyc);
    math("tr-t0", "\\tau = 0", tx0 - 8, ty, 11, C.muted, "right", a);
    math("tr-t1", "\\tau = 1", tx1 + 8, ty, 11, C.muted, "left", a);
    math("tr-zt", "Z_\\tau = (1 - \\tau)\\,\\varepsilon + \\tau Z, \\quad \\varepsilon \\sim \\mathcal{N}(0, I)", (x0 + x1) / 2, 360, 12, C.muted, "center", a);

    // DiT input tokens (as in Fig. 2): noisy plan + flow time, state, vision, steering, registers
    const groups = [
      { n: 4, c: C.grasp, t: "noisy plan", m: "Z_\\tau" },
      { n: 1, c: "#8a9aa6", t: "flow time", m: "\\tau" },
      { n: 1, c: C.ink, t: "state", m: "p_t" },
      { n: 3, c: "#a4623e", t: "vision", m: "d_t" },
      { n: 2, c: C.door, t: "steering", m: "c = E_{\\mathrm{steer}}(u)", w: 92 },
      { n: 2, c: null, t: "registers", m: null },
    ];
    const CW = 15, CG = 3, SP = 12, cy = 300;
    groups.forEach((g) => (g.sw = Math.max(g.n * CW + (g.n - 1) * CG, g.w || 60)));
    const total = groups.reduce((acc, g) => acc + g.sw, 0) + SP * (groups.length - 1);
    let gx = pl.x + pl.w / 2 - total / 2;
    groups.forEach((g, i) => {
      const on = a * ease((u - 0.06 - i * 0.05) / 0.14), cx = gx + g.sw / 2, cw = g.n * CW + (g.n - 1) * CG;
      for (let k = 0; k < g.n; k++) {
        const x = cx - cw / 2 + k * (CW + CG);
        ctx.save(); ctx.globalAlpha = on; roundRect(x, cy - CW / 2, CW, CW, 4);
        if (g.c) { ctx.fillStyle = g.c; ctx.globalAlpha = on * 0.85; ctx.fill(); }
        else { ctx.setLineDash([2, 2]); ctx.strokeStyle = C.muted; ctx.lineWidth = 1; ctx.stroke(); }
        ctx.restore();
      }
      label(g.t, cx, cy + 22, 10, C.muted, "center", 500, on);
      if (g.m) math(`tr-g${i}`, g.m, cx, cy + 39, 11.5, g.c || C.muted, "center", on);
      // into the DiT
      const top = { x: cx, y: cy - CW / 2 - 4 }, dst = { x: lerp(pl.x + 28, pl.x + pl.w - 28, i / (groups.length - 1)), y: pl.y + pl.h + 4 };
      const mid = (top.y + dst.y) / 2;
      curve(top, { x: top.x, y: mid }, { x: dst.x, y: mid }, dst, C.faint, on, 1.2);
      arrowHead(dst.x, dst.y - 1, -Math.PI / 2, C.faint, on, 6);
      stream(top, { x: top.x, y: mid }, { x: dst.x, y: mid }, dst, g.c || C.muted, on * 0.9, 2, 1.8, i * 0.17, 0.6);
      g.cx = cx; gx += g.sw + SP;
    });
    // the noisy plan is what gets tokenized
    const g0 = groups[0];
    curve({ x: x1 + 10, y: yc + 10 }, { x: x1 + 50, y: yc + 20 }, { x: g0.cx - 30, y: cy - 40 }, { x: g0.cx - 16, y: cy - 14 }, C.grasp, a * 0.55, 1.2, [3, 3]);
    // predicted velocity is regressed onto Z - eps
    curve({ x: pl.x + pl.w / 2, y: pl.y - 4 }, { x: pl.x + pl.w / 2, y: pl.y - 14 }, { x: pl.x + pl.w / 2, y: pl.y - 20 }, { x: pl.x + pl.w / 2, y: pl.y - 26 }, C.grasp, a * 0.7, 1.3);
    arrowHead(pl.x + pl.w / 2, pl.y - 28, -Math.PI / 2, C.grasp, a * 0.8, 6);
    math("tr-loss", "\\mathcal{L}_{\\mathrm{FM}} = \\mathbb{E}\\Big[\\big\\lVert v_\\theta(Z_\\tau, \\tau \\mid o, c) - (Z - \\varepsilon) \\big\\rVert_2^2\\Big]", pl.x + pl.w / 2, pl.y - 50, 13.5, C.ink, "center", a);
  }

  // ---- scene 4: real planner inputs from the pickup rollout -------------
  function drawInputs(a, u) {
    if (a <= 0.02) return;
    const pl = S.pl, plC = { x: pl.x + pl.w / 2, y: pl.y + pl.h / 2 };
    const X = 150, Wd = 144, Hd = 81;
    const rows = [{ y: 104, h: Hd }, { y: 214, h: 50 }, { y: 302, h: 34 }];
    const titles_ = [["Onboard depth", "d_t"], ["Proprioception", "p_t"], ["Steering token", "c_t"]];
    rows.forEach((r, i) => labelMath(`in-${i}`, titles_[i][0], titles_[i][1], X, r.y - 11, 11.5, C.ink, "left", 600, a, 4));
    label("language or waypoint", X, rows[2].y + rows[2].h + 14, 10, C.muted, "left", 500, a);
    // depth frames
    const m = inputs.meta, im = inputs.depth;
    ctx.save(); ctx.globalAlpha = a;
    roundRect(X, rows[0].y, Wd, Hd, 6); ctx.clip();
    // depth and joint traces replay the same rollout, once per scene, in sync
    const prog = clamp01((u * DUR[4] - 0.4) / 3.6);
    if (m && im.complete && im.naturalWidth) {
      const d = m.depth, f = prog * (d.frames - 1), i0 = Math.floor(f), fr = f - i0;
      const blit = (i) => ctx.drawImage(im, (i % d.cols) * d.fw, Math.floor(i / d.cols) * d.fh, d.fw, d.fh, X, rows[0].y, Wd, Hd);
      blit(i0); if (fr > 0.02 && i0 + 1 < d.frames) { ctx.globalAlpha = a * fr; blit(i0 + 1); }
    } else { ctx.fillStyle = "#56708a"; ctx.fillRect(X, rows[0].y, Wd, Hd); }
    ctx.restore();
    ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = C.panelEdge; roundRect(X, rows[0].y, Wd, Hd, 6); ctx.stroke(); ctx.restore();
    // joint traces (sliding window)
    ctx.save(); ctx.globalAlpha = a; roundRect(X, rows[1].y, Wd, rows[1].h, 6); ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.fill(); ctx.strokeStyle = C.panelEdge; ctx.stroke(); ctx.restore();
    if (m && m.proprio) {
      const ser = Object.values(m.proprio), cols = [C.door, C.grasp, C.pickup, C.muted];
      ser.forEach((s, j) => {
        const n = s.length - 1, end = Math.max(1e-3, prog * n);
        const at = (f) => { const i0 = Math.min(n - 1, Math.floor(f)); return lerp(s[i0], s[i0 + 1], f - i0); };
        const pt = (f) => ({ x: X + 6 + (f / n) * (Wd - 12), y: rows[1].y + rows[1].h - 7 - at(f) * (rows[1].h - 14) });
        ctx.save(); ctx.globalAlpha = a * 0.9; ctx.strokeStyle = cols[j % cols.length]; ctx.lineWidth = 1.3; ctx.beginPath();
        for (let f = 0; f < end; f += 0.5) { const p = pt(f); f ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y); }
        const e = pt(end); ctx.lineTo(e.x, e.y); ctx.stroke(); ctx.restore();
        if (prog < 1) dot(e.x, e.y, 1.8, cols[j % cols.length], a, false);
      });
    }
    // steering token pill
    const pw = 126, py = rows[2].y;
    ctx.save(); ctx.globalAlpha = a; roundRect(X, py, pw, rows[2].h, 17); ctx.fillStyle = "#eaf1f8"; ctx.fill(); ctx.strokeStyle = "#c4d4e6"; ctx.stroke(); ctx.restore();
    label("“pick up the box”", X + pw / 2, py + rows[2].h / 2, 11.5, C.door, "center", 600, a, true);
    // streams into the planner
    [rows[0].y + Hd / 2, rows[1].y + rows[1].h / 2, py + rows[2].h / 2].forEach((y, i) => {
      const x0 = X + (i === 2 ? pw : Wd) + 8;
      stream({ x: x0, y }, { x: x0 + 60, y }, { x: pl.x - 60, y: plC.y }, { x: pl.x - 2, y: plC.y }, C.door, a, 5, 2.2, i * 0.21);
    });
  }

  // ---- draw ------------------------------------------------------------
  function draw(dt) {
    const phase = phaseOf(clock);
    const u = reduced.matches ? 0.95 : sceneU(phase);   // reduced motion: each scene's completed state
    const loopN = playing ? 0 : Math.floor((animClock - enteredAt) / DUR[phase]);
    const cutKey = `${phase}:${enteredAt}:${loopN}`;
    if (cutKey !== lastCut) {
      lastCut = cutKey;
      const t0 = animClock + 0.3, rate = phase === 1 ? 0.6 : 1;
      [0, 1, 2].forEach((i) => clipRestart(`ex${i}`, t0 + (phase === 1 ? 0 : i * 0.25), rate));
      clipRestart("exe", t0);
    }
    if (reduced.matches) { const t = targets(phase); Object.assign(S, JSON.parse(JSON.stringify(t))); }
    else smoothAll(S, V, targets(phase), dt, ease((animClock - enteredAt - 0.22) / 0.35));
    const local = clock - START[phase];
    TXT = playing && !reduced.matches ? ease(Math.min(1, local / 0.45, (DUR[phase] - local) / 0.35)) : 1;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const s = vs * dpr;
    ctx.setTransform(s, 0, 0, s, vox * dpr, voy * dpr);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";

    const latC = { x: S.lat.x + S.lat.w / 2, y: S.lat.y + S.lat.h / 2 };
    const plC = { x: S.pl.x + S.pl.w / 2, y: S.pl.y + S.pl.h / 2 };
    const bfmC = { x: S.bfm.x + S.bfm.w / 2, y: S.bfm.y + S.bfm.h / 2 };

    const grow = phase === 2 ? [0, 1, 2].map((i) => ((v) => 1 - (1 - v) * (1 - v))(clamp01((u - 0.1 - i * 0.07) / 0.66))) : null;
    if (S.lat.a > 0.01) {
      ctx.save(); ctx.globalAlpha = S.lat.a;
      roundRect(S.lat.x, S.lat.y, S.lat.w, S.lat.h, 18); ctx.fillStyle = C.latentFill; ctx.fill();
      ctx.lineWidth = 1.3; ctx.strokeStyle = C.panelEdge; ctx.stroke(); ctx.restore();
      ctx.save(); roundRect(S.lat.x, S.lat.y, S.lat.w, S.lat.h, 18); ctx.clip();
      const g = ctx.createRadialGradient(latC.x, latC.y, 8, latC.x, latC.y, S.lat.w * 0.6);
      g.addColorStop(0, `rgba(120,180,175,${0.2 * S.lat.a})`); g.addColorStop(1, "rgba(120,180,175,0)");
      ctx.fillStyle = g; ctx.fillRect(S.lat.x, S.lat.y, S.lat.w, S.lat.h);
      fibers(S.lat.x, S.lat.y, S.lat.w, S.lat.h, S.lat.a, 0.55 + 0.45 * S.ov.collect + 0.4 * S.ov.train, S.ov.train > 0.3 ? 2 : -1, grow);
      ctx.restore();
      label("Shared executable latent space", latC.x, S.lat.y + 20, 14, C.ink, "center", 650, S.lat.a);
      labelMath("lat-sub", "frozen BFM ·", "z \\in \\mathbb{S}^{d-1}", latC.x, S.lat.y + 38, 11, C.muted, "center", 500, S.lat.a, 4);
    }
    if (S.ov.collect > 0.02) {
      for (let i = 0; i < 3; i++) {
        const e = S.ex[i];
        const p0 = { x: 242, y: e.fy - e.h * 0.42 };
        const p1 = fiberPath(fiberDefs[i], S.lat.x, S.lat.y, S.lat.w, S.lat.h)(0);   // enters its own fiber
        stream(p0, { x: p0.x + 70, y: p0.y }, { x: p1.x - 80, y: p1.y }, p1, expertCols[i], S.ov.collect, 7, 2.5, i * 0.13);
      }
      labelMath("co-leg", "closed-loop latent rollouts", "\\mathcal{D}^Z_i = \\{(d_t, p_t, u_t, z_t)\\}", latC.x, S.lat.y + S.lat.h + 22, 11.5, C.muted, "center", 500, S.ov.collect, 5);
    }

    box(S.pl.x, S.pl.y, S.pl.w, S.pl.h, S.pl.a, "DiT flow planner", "v_\\theta(Z_\\tau, \\tau \\mid d_t, p_t, c)", S.ov.plan > 0.5 || S.ov.pred > 0.5 || S.ov.train > 0.5 ? C.grasp : C.panelEdge, false, "pl");

    if (S.ov.train > 0.02) drawTrain(S.ov.train, phase === 3 ? u : 0.5, S.pl);
    if (S.ov.inputs > 0.02) drawInputs(S.ov.inputs, phase === 4 ? u : 1);
    if (S.ov.pred > 0.02) {
      const p0 = { x: S.pl.x + S.pl.w, y: plC.y };
      const path = (t) => ({ x: p0.x + 44 + t * 330, y: plC.y + Math.sin(t * Math.PI * 2 + 0.4) * 40 });
      // sampling: 10 Heun steps integrate the learned flow from noise to the plan
      const NS = 10, su = phase === 4 ? clamp01((u - 0.16) / 0.6) * NS : NS;
      const si = Math.min(NS - 1, Math.floor(su)), sp = su >= NS ? 1 : (si + ease((su - si) / 0.7)) / NS;
      const done = clamp01((sp - 0.9) / 0.1);
      curve({ x: p0.x + 4, y: plC.y }, { x: p0.x + 16, y: plC.y }, { x: p0.x + 26, y: plC.y }, { x: p0.x + 34, y: plC.y }, C.door, S.ov.pred * 0.6, 1.3);
      arrowHead(p0.x + 36, plC.y, 0, C.door, S.ov.pred * 0.8);
      ctx.save(); ctx.strokeStyle = C.door; ctx.globalAlpha = 0.25 * S.ov.pred * done; ctx.lineWidth = 1.4;
      ctx.beginPath(); for (let t = 0; t <= 1.0001; t += 0.02) { const p = path(t); t === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y); } ctx.stroke(); ctx.restore();
      const zl = ["z_t", "z_{t+3}", "z_{t+6}", "z_{t+H-1}"];
      for (let k = 0; k < 10; k++) {
        const p = path(k / 9), nx = p.x + (rand(k + 7) - 0.5) * 70 * (1 - sp), ny = p.y + (rand(k + 51) - 0.5) * 120 * (1 - sp);
        dot(nx, ny, 3, C.door, S.ov.pred * (0.35 + 0.65 * sp));
        if (k % 3 === 0) math(`pr-${k}`, zl[k / 3], p.x, p.y - 18, 12, C.door, "center", S.ov.pred * done);
      }
      // step pips
      const px0 = p0.x + 209 - 5 * 13 + 30;
      label("10 Heun steps", px0 - 10, plC.y - 92, 10.5, C.muted, "right", 500, S.ov.pred);
      for (let k = 0; k < NS; k++) {
        const on = sp >= (k + 1) / NS - 1e-3, cx = px0 + k * 13, cy = plC.y - 92;
        ctx.save(); ctx.globalAlpha = S.ov.pred * TXT; roundRect(cx, cy - 3, 10, 6, 3); ctx.fillStyle = on ? C.door : "#dbe3ea"; ctx.fill(); ctx.restore();
      }
      math("pr-noise", "\\varepsilon \\sim \\mathcal{N}(0, I)", p0.x + 209, plC.y + 84, 11.5, C.muted, "center", S.ov.pred * (1 - done));
      labelMath("pr-lbl", "latent plan", "\\hat{Z}_t = (z_t, \\dots, z_{t+H-1})", p0.x + 209, plC.y + 84, 12, C.door, "center", 600, S.ov.pred * done, 6);
    }

    box(S.bfm.x, S.bfm.y, S.bfm.w, S.bfm.h, S.bfm.a, "Frozen BFM", "\\pi_{\\mathrm{BFM}}(s_t, z_t)", null, true, "bfm");
    if (S.ov.plan > 0.02) {
      stream({ x: S.pl.x + S.pl.w, y: plC.y }, { x: S.pl.x + S.pl.w + 40, y: plC.y }, { x: S.bfm.x - 40, y: bfmC.y }, { x: S.bfm.x, y: bfmC.y }, C.door, S.ov.plan, 5, 2.6, 0, 0.35);
      math("ex-zk", "z_t", (S.pl.x + S.pl.w + S.bfm.x) / 2, bfmC.y - 20, 14, C.door, "center", S.ov.plan);
    }
    if (S.ov.exec > 0.02) {
      const hand = { x: S.exe.x - 58, y: S.exe.fy - S.exe.h * 0.6 };
      stream({ x: S.bfm.x + S.bfm.w, y: bfmC.y }, { x: S.bfm.x + S.bfm.w + 70, y: bfmC.y }, { x: hand.x - 80, y: hand.y }, hand, C.grasp, S.ov.exec, 9, 2.4, 0, 1.1);
      label("whole-body actions · 50 Hz", (S.bfm.x + S.bfm.w + hand.x) / 2, bfmC.y - 22, 11.5, C.grasp, "center", 600, S.ov.exec);
    }
    if (S.ov.feedback > 0.02) {
      const f = S.ov.feedback, ex = S.exe.x - 40, ey = S.exe.fy - 30;
      const c1 = [{ x: ex, y: ey }, { x: ex - 90, y: ey + 34 }, { x: bfmC.x + 60, y: S.bfm.y + S.bfm.h + 70 }, { x: bfmC.x, y: S.bfm.y + S.bfm.h + 6 }];
      curve(...c1, C.grasp, f * 0.7, 1.3, [4, 4]); arrowHead(bfmC.x, S.bfm.y + S.bfm.h + 5, -Math.PI / 2, C.grasp, f * 0.8);
      stream(...c1, C.grasp, f, 10, 1.9, 0, 1.5);                                  // every control step
      const m1 = bez(...c1, 0.55);
      labelMath("fb-hi", "high-rate state feedback", "s_t", m1.x + 14, m1.y - 16, 11, C.grasp, "left", 600, f, 5);
      const c2 = [{ x: ex - 10, y: ey + 16 }, { x: ex - 200, y: ey + 70 }, { x: plC.x + 40, y: S.pl.y + S.pl.h + 110 }, { x: plC.x, y: S.pl.y + S.pl.h + 6 }];
      curve(...c2, C.muted, f * 0.55, 1.2, [3, 5]); arrowHead(plC.x, S.pl.y + S.pl.h + 5, -Math.PI / 2, C.muted, f * 0.7);
      // one observation packet per replanning cycle; the planner flashes on arrival
      const RP = 1.8, tp = (animClock % RP) / 1.25;
      if (tp < 1) {
        for (let k = 0; k < 6; k++) { const q = bez(...c2, clamp01(tp - k * 0.018)); dot(q.x, q.y, 3.4 - k * 0.45, C.door, f * (1 - k / 6) * Math.min(1, tp * 8)); }
      } else {
        const fl = 1 - (tp - 1) / ((RP / 1.25) - 1);
        ctx.save(); ctx.globalAlpha = f * fl * 0.55; ctx.strokeStyle = C.door; ctx.lineWidth = 2 + 5 * (1 - fl);
        roundRect(S.pl.x - 3 - 6 * (1 - fl), S.pl.y - 3 - 6 * (1 - fl), S.pl.w + 6 + 12 * (1 - fl), S.pl.h + 6 + 12 * (1 - fl), 15); ctx.stroke(); ctx.restore();
      }
      const m2 = bez(...c2, 0.62);
      label("replanning at 5 Hz · depth, proprioception", m2.x, m2.y - 16, 11, C.muted, "center", 500, f, false, true);
    }

    for (let i = 0; i < 3; i++) {
      const e = S.ex[i];
      drawG1(`ex${i}`, expertKeys[i], e.x, e.fy, e.h, e.a, e.h < 150 ? 1.1 : 0);
      if (e.a > 0.05 && phase === 0) {
        label(expertNames[i], e.x, 430, 14, expertCols[i], "center", 600, e.a);
        label(["motion imitation", "staged reward shaping", "motion imitation"][i], e.x, 448, 11, C.muted, "center", 500, e.a);
      }
      if (e.a > 0.05 && phase === 2) label(expertNames[i], 238, e.fy - e.h * 0.72, 12, expertCols[i], "left", 600, e.a);
    }
    if (phase === 0) label("Heterogeneous interaction experts, acquired independently", VW / 2, 26, 12.5, C.muted);
    if (phase === 1) labelMath("lpp-expert", "Expert", "\\pi^{E}_i", 128, 426, 12.5, C.pickup, "center", 650, S.ex[0].a, 4);

    drawG1("exe", "pickup", S.exe.x, S.exe.fy, S.exe.h, S.exe.a);
    label("Unitree G1 · whole-body execution", S.exe.x, S.exe.fy + 22, 12, C.muted, "center", 500, S.exe.a);

    drawLPP(S.ov.lpp, phase === 1 ? u : 1);
    mathFrameEnd();
  }

  // ---- timeline --------------------------------------------------------
  let clock = 0, playing = !reduced.matches, visible = false, last = null, flow = 0, animClock = 0, dragging = false, TXT = 1;
  let enteredAt = 0, lastPhase = -1, lastCut = "";
  const phaseOf = (t) => { let p = 0; while (p < 5 && t >= START[p + 1]) p++; return p; };
  function sceneU(p) {
    if (p !== lastPhase) { lastPhase = p; enteredAt = animClock; }
    if (playing) return clamp01((clock - START[p]) / DUR[p]);
    return ((animClock - enteredAt) / DUR[p]) % 1;
  }
  // plain text with inline $...$ math typeset by KaTeX (re-typesets once KaTeX arrives)
  function renderRich(el, str) {
    const k = window.katex;
    if (el.dataset.src === str && (el.dataset.k === "1" || !k)) return;
    el.dataset.src = str; el.dataset.k = k ? "1" : "0";
    el.textContent = "";
    const host = el; el = document.createElement("span"); host.appendChild(el);   // one centered block
    str.split(/(\$[^$]+\$)/).forEach((part) => {
      if (!part) return;
      if (part[0] === "$" && part.length > 2) {
        const span = document.createElement("span");
        if (k) k.render(part.slice(1, -1), span, { throwOnError: false }); else span.textContent = part.slice(1, -1);
        el.appendChild(span);
      } else el.appendChild(document.createTextNode(part));
    });
  }
  function updateControls() {
    const phase = phaseOf(clock);
    root.dataset.stage = phase; root.dataset.playing = playing;
    if (phaseEl) phaseEl.textContent = titles[phase];
    if (detailEl) renderRich(detailEl, details[phase]);
    if (playBtn) playBtn.textContent = playing ? "Pause" : "Play";
    if (slider && !dragging) slider.value = Math.round((clock / TOTAL) * 1000);
    jumps.forEach((b, i) => b.setAttribute("aria-current", String(i === phase)));
  }
  let lastPhaseShown = -1;
  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible || document.hidden) { last = null; return; }
    const dt = last == null ? 16 : Math.min(now - last, 50); last = now;
    if (playing && !reduced.matches) {
      clock = (clock + dt / 1000) % TOTAL;
      const ph = phaseOf(clock); if (ph !== lastPhaseShown) { lastPhaseShown = ph; updateControls(); }
      else if (slider && !dragging) slider.value = Math.round((clock / TOTAL) * 1000);
    }
    flow = (flow + dt / 2400) % 1;
    if (!reduced.matches) animClock += dt / 1000;
    draw(dt);
  }
  const seek = (t) => { clock = Math.max(0, Math.min(TOTAL - 0.001, t)); last = null; lastPhaseShown = -1; updateControls(); };
  if (playBtn) playBtn.onclick = () => { playing = !playing; last = null; updateControls(); };
  if (replayBtn) replayBtn.onclick = () => { playing = !reduced.matches; seek(0); };
  if (slider) {
    slider.addEventListener("input", () => { dragging = true; playing = false; seek((Number(slider.value) / 1000) * TOTAL); });
    slider.addEventListener("change", () => { dragging = false; });
  }
  jumps.forEach((b) => (b.onclick = () => { playing = false; lastPhase = -1; seek(START[Number(b.dataset.fiberJump)] + 0.05); }));
  reduced.addEventListener("change", (e) => { if (e.matches) { playing = false; updateControls(); } });
  new ResizeObserver(() => { resize(); for (const e of mathPool.values()) e.w = 0; }).observe(stage);
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; last = null; if (visible) requestAssets(); }, { threshold: 0.05 }).observe(stage);
  new IntersectionObserver((es) => { if (es[0].isIntersecting) requestAssets(); }, { rootMargin: "900px 0px" }).observe(stage);
  resize(); updateControls();
  requestAnimationFrame(frame);
})();
