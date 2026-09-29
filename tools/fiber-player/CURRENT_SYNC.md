# Local checkout handoff — 2026-09-29

The user requested stopping local work and syncing this working tree to the L40S node.
This is an uncommitted working-tree snapshot, not a validated release or a GitHub push.

## Validated changes

- Local preview server: `python3 tools/fiber-player/serve_preview.py --port 8001`.
  COOP/COEP enables four WASM CPU planner threads. Non-isolated hosts retain one.
- Default remains W8A32 step-2000 distilled student, 3 Euler steps, synchronous
  1 Hz simulation cadence; FP32 actor. Physics waits for planner inference.
- Same-input four-run median planner timing: Chromium 1535 -> 546 ms;
  WebKit 1473 -> 475 ms. Body and hand tensors were identical between one
  and four threads for the captured door input in each engine.
- Overview now retains the three artwork image nodes, interpolates their
  transforms, updates diagram attributes in place, and crossfades diagram changes.
  Chromium/WebKit checks passed persistent image identity, rapid navigation,
  stable mobile figure height, and reduced motion.

## Door work INCOMPLETE / do not claim resolved

User reports failures in both Safari and Chrome after walking.
Cold and idle-three-seconds starts passed in both browser engines before changes.
A broader real-WASD test reproduced failures after forward walking and turning.
Facing/settling alone failed and was replaced with returning to the approach
position [0.15, -0.6] through the locomotion bank, then facing and settling.
The latest approach implementation is NOT validated:
- Latest Chromium sweep: cold pass, idle pass, forward FAIL, side pass (24.66 s),
  turned pass (15.70 s). Testing was stopped at the user's request before the
  sweep and WebKit validation finished.
- A final small change to approach arrival hysteresis was made after that browser
  had loaded, so those results do not validate that last edit either.
- `playground/player.js`: beginDoor, approach_door, settle_door, per-phase arrival.
- `playground/locomotion.js`: optional arrivalRadius (default .3 m, door .16 m).
- Open door continues to receive NO planner waypoints. Preparation is physical
  locomotion; no teleport/reset. State drops only on true executed command switches.
- No native FP32 or model/scene changes were made in this session.

Reports: `tools/fiber-player/sync-reports/2026-09-29/`.
Browser audit entry points: `audit_door_switches.cjs`, `audit_overview_transitions.cjs`.
Set PLAYWRIGHT_MODULE to the installed Playwright module if not on the Node path.
These are fixed-sampler checks, not population success-rate estimates.
