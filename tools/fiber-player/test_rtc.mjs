import assert from "node:assert/strict";
import { RtcBuffer } from "../../playground/rtc.js";
import { blend } from "../../playground/locomotion.js";
const controls = (offset = 0) =>
  Array.from({ length: 50 }, (_, i) => ({
    z: Float32Array.of(
      Math.SQRT2 * Math.cos(i / 100 + offset),
      Math.SQRT2 * Math.sin(i / 100 + offset),
    ),
    hand: Float32Array.of(i + offset * 100),
    command: "pickup",
  }));
const rtc = new RtcBuffer(blend);
rtc.reset(1, "pickup", 0);
const packet = (tick = 0) => ({
  version: 1,
  epoch: rtc.epoch,
  command: "pickup",
  tick,
  controls: controls(),
});
assert.equal(rtc.install(packet(), 10), "accepted");
assert.equal(rtc.queue.length, 40);
assert.equal(rtc.take().hand[0], 10); // Body and hand advance by the same inference delay.
for (let i = 0; i < 9; i++) rtc.take();
const old = rtc.queue[0].z.slice();
const next = { ...packet(15), controls: controls(1) };
assert.equal(rtc.install(next, 20), "accepted");
assert.deepEqual(rtc.queue[0].z, old); // First overlapping body control is continuous.
assert.equal(rtc.queue[0].hand[0], 105); // Hand targets are not crossfaded.
for (const c of rtc.queue)
  assert.ok(Math.abs(Math.hypot(...c.z) - Math.SQRT2) < 1e-6);
const saved = rtc.queue;
assert.equal(rtc.install({ ...packet(), version: 0 }, 20), "stale");
assert.equal(rtc.install({ ...packet(), epoch: rtc.epoch - 1 }, 20), "stale");
assert.equal(rtc.install({ ...packet(), command: "open_door" }, 20), "stale");
assert.equal(rtc.install(packet(), 50), "expired");
assert.equal(rtc.queue, saved);
rtc.reset(2, "walk", 0);
assert.equal(rtc.queue.length, 0);
rtc.requested(0);
assert.equal(rtc.nextRequestTick, 50);
assert.equal(rtc.due(49, false), false);
assert.equal(rtc.due(50, true), false);
rtc.requested(155);
assert.equal(rtc.nextRequestTick, 200);
console.log(
  "PASS RTC delay alignment, body continuity, hand hold, stale results, expiry, request cadence",
);
