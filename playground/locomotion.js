// Deployment navigation: continuously blend forward/turn skills on the z sphere.
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
export function blend(a, b, t) {
  t = clamp(t, 0, 1);
  const ra = Math.hypot(...a),
    rb = Math.hypot(...b);
  if (!ra || !rb) throw new Error("Invalid zero locomotion latent");
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += (a[i] / ra) * (b[i] / rb);
  dot = clamp(dot, -1, 1);
  if (dot > 0.9999) {
    const z = Float32Array.from(
      a,
      (v, i) => ((1 - t) * v) / ra + (t * b[i]) / rb,
    );
    const n = Math.hypot(...z);
    return Float32Array.from(z, (v) => (v * ra) / n);
  }
  if (dot < -0.9999) {
    // Opposite turn skills have no unique geodesic. Choose a stable orthogonal
    // direction rather than dividing by sin(pi) and losing the sphere radius.
    let axis = 0;
    for (let i = 1; i < a.length; i++)
      if (Math.abs(a[i]) < Math.abs(a[axis])) axis = i;
    const orthogonal = Float32Array.from(
      a,
      (v, i) => (i === axis ? 1 : 0) - (v * a[axis]) / (ra * ra),
    );
    const n = Math.hypot(...orthogonal);
    const middle = Float32Array.from(orthogonal, (v) => (v * ra) / n);
    return t < 0.5 ? blend(a, middle, t * 2) : blend(middle, b, t * 2 - 1);
  }
  const angle = Math.acos(dot),
    s = Math.sin(angle);
  const sa = Math.sin((1 - t) * angle) / s,
    sb = Math.sin(t * angle) / s;
  return Float32Array.from(
    a,
    (v, i) => ra * ((sa * v) / ra + (sb * b[i]) / rb),
  );
}
export function validateBank(bank, dimension) {
  for (const key of ["forward", "turn_left", "turn_right", "stand"]) {
    const z = bank[key];
    if (
      !Array.isArray(z) ||
      z.length !== dimension ||
      !z.every(Number.isFinite)
    )
      throw new Error(`Invalid locomotion bank: ${key}`);
    if (Math.abs(Math.hypot(...z) - Math.sqrt(dimension)) > 0.01)
      throw new Error("Locomotion bank radius mismatch");
  }
  return bank;
}
export function locomotionZ(
  bank,
  {
    x,
    y,
    yaw,
    goal,
    heading,
    yawRate = 1.8,
    arrived = false,
    arrivalRadius = 0.3,
    vx = 0,
    vy = 0,
    wz = 0,
  },
) {
  const dx = goal ? goal[0] - x : 0,
    dy = goal ? goal[1] - y : 0,
    dist = Math.hypot(dx, dy);
  const facing = dist <= arrivalRadius || arrived;
  const desired = facing ? (heading ?? yaw) : Math.atan2(dy, dx);
  const error = wrap(desired - yaw);
  // Account for the motion already under way, so braking starts before crossing
  // the target heading. Keep the forward/turn interpolation from deployment.
  const predicted = wrap(error - 0.24 * wz);
  const positive = bank.meta?.turn_left_for_positive_error ?? true;
  const turn = predicted > 0 === positive ? bank.turn_left : bank.turn_right;
  if (facing) {
    if (Math.abs(error) <= 0.1 || error * predicted <= 0)
      return Float32Array.from(bank.stand);
    // Native actor response has different stand→turn onset on the two sides.
    // Blend continuously through that onset; do not toggle full turn skills.
    const onset = turn === bank.turn_left ? 0.72 : 0.51;
    const e = Math.abs(predicted);
    const ramp = clamp(e / 0.12, 0, 1);
    const amount =
      (onset + (1 - onset) * clamp(e / ((0.65 * 1.8) / yawRate), 0, 1)) *
      ramp *
      ramp *
      (3 - 2 * ramp);
    return blend(bank.stand, turn, amount);
  }
  const amount =
    Math.abs(predicted) <= 0.05
      ? 0
      : clamp(Math.abs(predicted) / ((0.7 * 1.8) / yawRate), 0, 0.97);
  let z = blend(bank.forward, turn, amount);
  const closing = dist > 0 ? Math.max(0, (dx * vx + dy * vy) / dist) : 0;
  const remaining = Math.max(0, dist - 0.25 * closing);
  if (remaining < 1)
    z = blend(
      z,
      bank.stand,
      Math.min(0.85, 1 - remaining) * Math.max(0, 1 - amount / 0.65),
    );
  return z;
}
// Smooth every actor control and limit the geodesic rate, including task switches.
export function smoothLocomotionZ(previous, target, stand, dt = 0.02) {
  let dot = 0;
  const r = Math.hypot(...previous),
    s = Math.hypot(...target);
  for (let i = 0; i < previous.length; i++)
    dot += (previous[i] * target[i]) / (r * s);
  if (dot < -0.8) target = stand; // Reverse through neutral, not the antipodal turn pole.
  const angle = Math.acos(clamp(dot, -1, 1));
  return blend(
    previous,
    target,
    Math.min(1 - Math.exp(-dt / 0.18), (4 * dt) / Math.max(angle, 1e-6)),
  );
}
