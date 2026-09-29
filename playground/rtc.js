// Latency-aware execution, matching deploy/five_knot_runtime.py. Hands remain
// zero-order held: only the body latent is blended across overlapping plans.
export class RtcBuffer {
  constructor(blend, interval = 50, ramp = 5) {
    this.blend = blend;
    this.interval = interval;
    this.ramp = ramp;
    this.epoch = 0;
    this.reset(-1, null, 0);
  }
  reset(version, command, tick) {
    this.epoch++;
    this.version = version;
    this.command = command;
    this.queue = [];
    this.nextRequestTick = tick;
    this.accepted = this.expired = this.underruns = this.delay = 0;
  }
  due(tick, pending) {
    return !pending && tick >= this.nextRequestTick;
  }
  requested(tick) {
    const missed = Math.max(
      0,
      Math.floor((tick - this.nextRequestTick) / this.interval),
    );
    this.nextRequestTick += (missed + 1) * this.interval;
  }
  install(packet, tick) {
    if (
      packet.version !== this.version ||
      packet.epoch !== this.epoch ||
      packet.command !== this.command
    )
      return "stale";
    this.delay = Math.max(0, tick - packet.tick);
    if (this.delay >= packet.controls.length) {
      this.expired++;
      return "expired";
    }
    const next = packet.controls.slice(this.delay);
    const overlap = Math.min(this.ramp, this.queue.length, next.length);
    for (let i = 0; i < overlap; i++) {
      next[i] = {
        ...next[i],
        z: this.blend(this.queue[i].z, next[i].z, i / Math.max(1, overlap - 1)),
      };
    }
    this.queue = next;
    this.accepted++;
    return "accepted";
  }
  take() {
    return this.queue.shift() || null;
  }
}
