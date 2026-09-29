# FIBER default-success audit: on-node handoff

## Objective

Verify a real default success for door opening/crossing, box pickup from ground and supports, carrying with waypoint control, and separate Tangerine / Water bottle / Cup grasp scenes. Grasp resets must have arms down and all 14 Dex3 finger joints open. Each grasp scene contains exactly one object. Native success is a screening result; the browser's exported-model success is a separate required gate.

## Environment

SSH from the laptop:

```sh
ssh -i /Users/yutanglin/Downloads/qiyangy-ec2-1.key ubuntu@43.203.203.30
cd /home/ubuntu/fiber-web-export
export PYTHONPATH=src
export MUJOCO_GL=egl
```

Use `venv/bin/python`. Node has an NVIDIA L40S and MuJoCo 3.14.0. Check installed versions/providers before running. Do not replace this with old MuJoCo 3.3.7: previous grasp behavior differed.

```sh
venv/bin/python -c 'import torch,mujoco,onnxruntime as o; print(torch.__version__,mujoco.__version__,o.__version__,o.get_available_providers())'
```

Model files remain on node:

- `models/planner.pt`: latest v27 planner, step 226000, body + Dex3 head.
- `models/model_best_emd.pt`: BFM body controller.
- `models/language.pt`: v5 language cache. Do not use `language-v3.pt` for grasp.
- `browser/actor.onnx`: FP32 actor.
- `browser/v27/planner.{condition,time,step.gpu}.w8.onnx`: INT8 weight storage, FP32 activations.
- `browser/v27/planner_meta.json` and `browser/actor_meta.json`: exact normalization, hand target mapping, prompts, solver contracts.
- `browser/scenes/grasp/objects.json`: three single-object scenes; configs are `candidate-tangerine.json`, `candidate-water-bottle.json`, `candidate-cup.json`.
- `browser/scenes/pickup/layouts.json`: mixed / ground / support layouts, three selectable boxes per layout.
- `browser/scenes/door/scene.json`: door default.

## Known results — do not conflate the two runtimes

Native PyTorch FP32 seed-search results, using 8 Euler steps and one state-dropped first interaction plan, from the neutral reset with original planner finger timing:

| Scene        | Seed |    Lift | Successful hold/contact |
| ------------ | ---: | ------: | ----------------------: |
| Tangerine    |   14 | 0.297 m |         51 / 51 samples |
| Tangerine    |    5 | 0.216 m |         51 / 51 samples |
| Water bottle |    4 | 0.322 m |         51 / 51 samples |
| Water bottle |   19 | 0.292 m |         51 / 51 samples |
| Cup          |   13 | 0.268 m |         51 / 51 samples |

Logs: `single-search-fixed.log`, `water-search.log`, `water-search-more.log`, `cup-search.log`.

Actual local WebGPU / W8 planner checks:

- Door opened and crossed: 9.54 sim seconds, hinge 1.345 rad, robot x 1.805 m.
- First supported box: 2.62 seconds, lift 0.396 m, held 51 samples, contact 45/51.
- Tangerine seed 14, original finger timing: **passes** at 7.68 seconds, lift 0.245 m, contact 51/51.
- Water bottle seed 4: **fails** to lift by 20 seconds.
- Cup seed 13: **fails** to lift by 20 seconds.
- Remaining box selections, full chain, and all carry destinations are not fully verified.

A delayed finger-closure experiment passed native Tangerine seed 43, but the selected default now uses the original planner timing and seed 14. Do not silently reintroduce delayed closure or the reference-reach experiments.

## First commands: reproduce native results

`fiber-real-search.py` is a native FP32 batched screening harness, not an ONNX/browser evaluator. It runs seeds in a batch of 16 starting with the second argument and reports successful physical lifts with contact. It reads the current single-object config for its first argument.

```sh
PYTHONPATH=src MUJOCO_GL=egl venv/bin/python fiber-real-search.py tangerine 0 > audit-tangerine.log 2>&1
PYTHONPATH=src MUJOCO_GL=egl venv/bin/python fiber-real-search.py water-bottle 0 > audit-water-bottle.log 2>&1
PYTHONPATH=src MUJOCO_GL=egl venv/bin/python fiber-real-search.py cup 0 > audit-cup.log 2>&1
```

`real-search-more.py water-bottle 16` screened later seeds and found 19. It stops after the first successful batch. Inspect its loop before using it as a claim about a larger seed range.

Next, rerun each winning seed individually with the same initial config and a strict success gate. Record the full initial qpos/qvel and configuration hash. Batched screening proves a candidate; an individual rerun and exported-model check are still required.

## Exact inference and physics contract

Match the browser reference, rather than an older evaluation preset:

- Physics dt and PD gains/effort limits/decimation come from scene JSON. Current control runs at 50 Hz with 20 substeps of 0.001 s.
- Body joint order comes from scene `qadr/vadr` and actor metadata. Never substitute Isaac's interleaved joint order.
- Actor observation: 64 state values, last action, and 9 lagged frames in term-major order: actions, angular velocity, joint-position deviation, joint velocity, gravity. History starts with zero lag frames and the actual current frame.
- Actor action: clamp `actor_output * 6` to [-6,6]; target = `default_q + action * 0.25 * action_rescale`; apply bounded PD torques. Do not treat actor output as direct joint targets.
- Native camera is `camera_id` from config, 426x240. Resize nearest to 256x144, round to millimeters, then convert to meters; retain only [0.12,3] m.
- Planner state normalization uses `normalize_planner_state` / metadata. Actor and planner state are different inputs.
- Hand state = `(actual 14 joint positions - mean) / std` from hand normalization metadata.
- Hand target = `center + clip(prediction,-1,1) * scale`, with 5 controls per hand knot, using the scene's real finger PD actuators.
- Precision tasks use 8 Euler steps. Cached browser timestep layers must use the corresponding eight times. Four-step manual walk is a separate path.
- RTC: 10 knots, 5 controls/knot, 25 controls/replan, first 5 control blend, control_points=2. Anchor from actual BFM backward encoding at reset. Never advance the execution anchor into unexecuted future controls.
- Drop only the planner state token for exactly the first accepted grasp interaction plan. Pickup now retains state after a verified FP32 regression from the one-plan drop. Actor proprioception stays present. Resume planner state immediately thereafter. Door/walk/carry currently retain state; dropping on door switches broke the tested default.
- Grasp language is `grasp_table`; no waypoint command is conditioned for grasp. `hand_mask=True`.
- Export Gaussian sampler inputs for the exact seed using CUDA Torch, drawing body `[1,10,512]`, then hand `[1,10,14]` for each plan. Matching a seed number with NumPy noise is insufficient.

## Required success gates

At 50 Hz, for pickup/grasp require all of:

1. Object center rises >0.10 m from reset.
2. Object remains within 0.80 m horizontally of pelvis.
3. At least 51 consecutive raised/near samples (one second).
4. At least 45 of those last 51 samples have actual hand/wrist-to-selected-object normal contact force >0.1 N, obtained with `mj_contactForce`.
5. Robot remains standing; no fall or unsupported object launch.

For door: hinge >0.5 rad **and** pelvis crosses the doorway (current default x >1.6 m), without a fall. For carry: continue after confirmed pickup, reach an edited waypoint/heading while retaining lift and hand contact; do not count pickup alone as carrying success.

Capture JSON results and a TPV + camera/depth video or key frames. Include seed, model hashes/precision, scene hash, solver/RTC settings, state-drop plan count, sim time, max lift, hold/contact counts, root pose, hinge where relevant, and fall/timeout reason. These are curated demo defaults, not a newly measured paper success rate.

## Exported-model gate: highest priority unresolved issue

Run the exact browser ONNX graphs on the node using ONNX Runtime first. CPU provider is acceptable for correctness; check CUDA provider availability rather than assuming it exists. Reproduce the condition/time/step caching and fused Euler update from `browser-reference/planner-worker.js` and the actor/observation/scheduler path from `browser-reference/player.js`.

For each native winning seed, compare:

1. PyTorch FP32 planner + PyTorch actor.
2. W8 split ONNX planner + FP32 ONNX actor, same scene/camera/noise/physics.
3. Actual browser WebGPU on the laptop (final integration gate).

Save first-plan body/hand endpoint errors and executed per-control trajectories. This distinguishes quantization differences from renderer/normalization/scheduler differences. If W8 fails while FP32 passes, test selective higher-precision weights or FP16 weight storage with FP32 computation **on node first**. That higher-precision export has been proposed but is not implemented yet. Do not change the actor to FP16; an earlier actor FP16 test produced invalid outputs.

## Remaining boxes

`pickup-approach-fix.py` is an experiment, not current website behavior. It uses a closer 0.35 m standoff and 0.28 m transition gate; the website currently has 0.45 m / 0.18 m. `pickup-approach-fix.log` shows that increasing the gate lets ground box 2 start pickup, but does not establish a full hold for all boxes. Inspect final mode and route/obstacle collisions before tuning pickup itself. Audit every layout/object pair; do not generalize the first-box result to all nine cases.

## Constraints

Keep exactly one object per grasp scene; arms down/open fingers at reset. No pre-grasp reset, live qpos teleport, scripted lift, object forces, welding, fake contacts, or success thresholds weakened to turn failures into passes. Do not transfer large training checkpoints to the laptop. Keep experiments separate from verified defaults. No website push/deployment; local preview only.

Return: reproducible commands, machine-readable results, videos/key frames, exact verified configs/seeds, and compact inference exports only if they improve the exported-model gate.

## New box FP32 audit (latest)

`audit_pickup_native.py` now provides explicit `--layout`, `--object`, `--seed`,
`--retain-state`, `--isolate`, `--standoff`, `--approach-gate`, and `--face-object`
controls. The current version checks approach transitions after every executed
control, and requires the actual pickup phase as well as physical lift/contact
for success. Older `pickup-native-baseline.jsonl` approach cases checked handoff
only at plan boundaries; ground object 3 physically lifted in walk mode, so its
reported success is not a verified UI pickup completion.

Native first box with retained state: mixed/support complete at 2.28 s, ground
at 2.10 s. Local W8 browser reconfirmed mixed (lift .404 m, contact 51/51) and
ground (lift .689 m, contact 51/51). The website now retains state for pickup.

`pickup-native-isolated.jsonl`: 8/9 box layout/object combinations pass native
FP32 with state retained when the chosen box is isolated and the robot starts
at the original arrival pose relative to it. The low support (.25 m), support
object 2, still fails at seed 42. This isolation is an audit reset, not a live
robot teleport or an implemented website scene change. Remaining multi-box
walking approaches are not solved by the first-box fix.

## Latest user correction: state drop on command switches only

The browser now retains planner state on the first plan after reset, including
grasp. It drops state for exactly one plan when changing from the last actually
executed model command during an existing simulation. The next plan resumes
state. Repeated commands, goal edits, and phase/status changes do not trigger
dropout. Held-object carrying still retains state.

Earlier successful grasp seed results above used a state-dropped first plan;
they are historical candidates and must be revalidated under this corrected
startup rule. Update native evaluators to use the same executed-command tracking;
do not enable dropout merely because the first command was selected after reset.

## Current browser default fixes (2026-09-27)

Original multi-box UI default sweep passed 5/13: pickup 3/9, door 1/1, grasp 1/3. Pickup selectors now load separate one-box scenes for all nine placements. Export with `export_pickup_single_cases.py`, followed by `export_pickup_low_candidates.py`; low support uses episode 14 arrival and original support/box geometry. The first eight single-box cases and the replacement low-support case have passed actual Chromium Metal browser contact/hold gates. Native passage is not substituted for browser evidence.

Neutral grasp sampler seeds with startup state retained: Tangerine 15 (browser pass at 8.24s), Water bottle 4 (browser pass at 12.46s), Cup 52 (browser pass at 9.92s). Each scene has one object, arms down and all fingers open. No live pose overrides or extra object forces. `audit_browser_defaults.cjs` produces JSONL and screenshots from actual UI interactions.

Final browser default set passed 13/13: pickup 9/9, door 1/1, grasp 3/3. Cup seeds 9, 21 and 37 failed W8 despite passing native FP32; seed 52 passed browser from the original neutral tabletop scene. Fixed default trials are not randomized success-rate evaluation.

## Pickup sizes, randomized starts, and 3D waypoints (2026-09-27)

After regenerating the isolated pickup scenes, run `resize_pickup_cases.py` on the
scene folder. It updates XML collision geometry, visual sizes, and all nine case
records together. Small/medium/large full dimensions are 17×16×21.5,
18.5×27.2×27.5, and 24×32×34 cm. Mass stays at the original 0.5 kg.

Pickup reset samples a robot position 0.9–1.5 m from the selected box, with
±65° side variation and ±60° heading variation around the front approach sector.
The original root tilt, joint pose, and box/support placement are retained.
The Pickup button directly selects the pickup language prompt with no waypoint
and command conditioning masked out. It does not insert an automatic walking
phase. Users can select Walk to approach before requesting Pickup. Carry remains
enabled after load/reset and can be selected without a confirmed held box.
State is dropped only on a true command switch after actual execution; startup
retains it. Run reports record the sampled spawn and phase transitions.

Walk and Carry expose body height (0.40–1.00 m, default 0.75 m). This is the
planner's absolute pelvis-height target in each `xy_yaw_height` command token.
It does not command a footstep or object height. Height edits are smoothed with
the existing 0.14 s waypoint filter and do not drop state. Conditioned waypoint
markers use the actual height token; manual goals show a vertical stem.

Browser audit `audit_browser_defaults.cjs` now checks pickup spawn bounds and
compiled collision dimensions and records initial state plus accepted plan events.
Set `PICKUP_ONLY=1` to test all nine selectable pickup setups.

## Continuous runs and locomotion bank (2026-09-27)

Task completion, physical falls, and time limits do not automatically pause.
Reached manual waypoints retain position, heading, and body-height targets.
Run/Pause is explicit; running scenes resume after reset or scene/object reload.
Inference exceptions or non-finite physics remain runtime errors.

Ordinary Walk at the default 0.75 m body-height target overrides the planner
with the deployment navigation z-bank and runs the frozen FP32 actor at 50 Hz.
The bank is rebuilt for BFM-0.5 iteration 75275, 256-dimensional latents, profile
`bfm05_xl`; the existing deployment bank was 512-dimensional and incompatible.
Exported JSON is 46,097 bytes and includes the exact actor ONNX SHA-256.
Native response calibration measured forward 0.629 m/s and positive turn
3.191 rad/s. Heading feedback blends forward/turn skills and decelerates near
the waypoint. It enters final-facing within 0.3 m and returns to approach if
turning drift exceeds 0.5 m. Latent changes use 0.12 s smoothing.

Carry and interactions retain the planner to preserve their body/object pose.
Height-adjusted Walk also retains the planner because this bank has no height
conditioning. Both paths retain actual executed-command tracking and the
existing rule for state dropout on a genuine command switch.

Build on the node with `build_deploy_zbank.py --checkpoint models/model_best_emd.pt
--out browser/locomotion --episodes 32 --steps 200 --calib-basis`. Transfer only
`z_bank.json`; keep checkpoints remote. The browser checks dimension and sphere
radius. Direct navigation displays the live goal, not unused planner horizon
tokens.

Explicit Walk selection now always chooses `walk` and resets body-height
target/filter to 0.75 m so the bank takes over immediately. It clears stale
planner continuation and navigation arrival state. Holding/lift detection does
not silently change Walk into Carry. Carry selection explicitly chooses
`carry_locomotion`; changing height during Walk still uses the height-conditioned
planner path.

## Large-box modes and browser checks (2026-09-27, latest)

The local preview uses Ground, Support, and Throw modes with the same large
24 × 32 × 34 cm box. Support uses its 50 × 60 cm platform with a 0.55 m top.
Pickup starts at 0.9–1.5 m, in a varied side sector, with heading offsets limited
to ±20 degrees to keep the object visible. Pickup and Door have no conditioned
waypoints. Carry remains selectable. Throw adds a user-directed physical
velocity impulse after a mouse drag; it does not reposition or attach the box.

Default model: FP32 actor, W8A32 v27 teacher planner, 8 Euler steps.
The three-step student remains accessible with `?planner=student`.
F16 weight storage is an optional experiment, not the default.

Large-box WebGPU checks: Chromium and isolated WebKit each passed 2/2 Ground
and 2/2 Support starts, with physical sustained-contact gates. Reports are in
`Artifacts/browser-reports/2026-09-27-large-box-modes` outside the website repo.
These are small regression checks, not measured aggregate success rates.
Isolated WebKit is not native Safari. A forced-WASM Support check passed its
first randomized start at 5.70 s (radius 1.43 m); second trial failed at 20.12 s (radius 1.21 m). CPU Support is not yet reliable.

Locomotion smoothing is now 0.18 s with a 4 rad/s geodesic rate limit and
predictive braking. Short and long arrival checks settled in Chromium and
WebKit. The newer local deployment code uses directional neutral offsets plus
one sphere projection. Two rebuilt directional banks had unusable positive
yaw endpoints (~0.05 rad/s), so neither replaces the validated existing bank.

Canonical relaxed-arm grasp initialization was tested as `?graspStart=relaxed`.
Cup/8 steps, Tangerine/4, and Bottle/16 all failed in isolated WebKit WebGPU
at 18 s; originals remain default. Cross-backend grasp reliability remains
unresolved. Do not describe all three grasp objects as verified successful.

Local only; no commit, push, or publication in this work.

### Support orientation correction (2026-09-28)

Support now rotates the initial box quaternion with the randomized root heading,
matching Ground and preserving the demonstrated relative grasp face. The large
platform contains the rotated box footprint; neither collision sizes nor
success/contact thresholds changed. Both randomized W8/8-step forced-WASM
Support trials passed (5.38 s and 5.60 s). The previously failed exact start still failed at 20.04 s after alignment: radius
1.2082869703 m, side 0.0411095292 rad, heading offset -0.0694504536 rad.
A same-start FP16-storage comparison is in progress. F16-storage CPU Support also passed two starts,
but different random consumption produced different second starts, so this
does not isolate quantization causality and the default remains W8.

Large-box Throw was verified with a real Chromium mouse drag: physical velocity
changed and the scene continued under Pickup without page errors.

Further checks: the same failed CPU Support start also failed with FP16 weight
storage at 20.06 s. A W8 16-step replay hit the audit's 240 s wall-clock
timeout and is inconclusive. The three-second neutral Run-before-Grasp sequence
failed all three objects in isolated WebKit WebGPU at 23 s total simulation time.
Initial Cup depth and qpos were identical between Chromium and isolated WebKit
(all 36,864 depth values exactly equal); this only verifies the reset input,
not subsequent closed-loop observations or inference parity.

Cup root XY [-0.06, 0.60] and CUDA noise seed 42 succeeded in isolated
WebKit WebGPU at 10.94 s, but failed in Chromium WebGPU and WebKit WASM.
The standard Chromium preset is therefore retained. Teacher/W8 WebKit WebGPU
now uses per-object measured presets: Tangerine 4 Euler steps; Bottle 16;
Cup 8 with the seed-42 pose above. These are explicit deployment presets, not
changes to physical success criteria. Full UI preset replay is in progress.
WebKit WASM retains standard presets; CPU grasp reliability remains unresolved.

Final UI replay of WebKit WebGPU presets: Cup/8 steps passed at 10.94 s;
Bottle/16 passed at 9.20 s; Tangerine/4 passed at 7.18 s. Each reached 51
sustained hand-contact frames. The trial helper's `trial.steps` field was an
unused label; actual executed steps are recorded in `solverSteps`. No audit
configuration or warm-up was applied in this replay. All start with resting
arms and open fingers. CPU Support and CPU grasp still have known failures.
Preview is local at http://localhost:8001/playground/.

## Default restored to distilled student (2026-09-28, latest)

At the user's request, the default is the run-2 step-2000 distilled student
`v27-distill2000-228000`, SHA-256
`d8441d6cdaf24d01e7a0ceb9fd309c31c5e42881fb55f0d193504986df56a483`,
with three Euler steps for every planner task. Actor stays FP32; planner is
W8A32. The teacher is only selected explicitly by `?planner=teacher` (or the
explicit F16 teacher storage experiment). Teacher-only WebKit grasp solver
presets do not apply to the default student. Previous teacher success results
are not evidence for this student configuration. Local UI checks in progress.

Student default UI checks in Chromium WebGPU: Ground passed at 5.22 s and
Support at 5.60 s, each with 51 sustained hand-contact frames. Saved reports
confirm the loaded model is `v27-distill2000`, actual solverSteps=3, FP32 actor
and W8A32 planner. One randomized start per mode was checked.

Isolated WebKit WebGPU student/3-step checks: Tangerine passed at 8.00 s
with 45 sustained hand-contact frames; Cup and Bottle failed at 20 s. Actual
model/solver were asserted from the report. Unused `trial.steps=8` labels in
the temporary helper are not executed overrides; `solverSteps=3` records the
actual inference setting. The student remains the default, without teacher
fallback. No publication or git commit/push.

## One-Hz planning trial (2026-09-28, latest)

The distilled student remains default with three Euler steps. Interaction
plans now execute 50 actor controls (one simulation second) instead of 25.
The ten-knot, five-controls-per-knot output already covers this full second;
body interpolation and zero-order-held Dex3 targets use the full planned
horizon. Plans start from the last executed latent. No extrapolation or
repetition of the final knot is introduced. Ordinary bank-controlled Walk
continues at 50 Hz. Command changes and live waypoint edits can interrupt
a plan early. Inference remains synchronous, so one Hz refers to simulation
time; wall-clock throughput also includes inference latency.

One-Hz Chromium WebGPU pickup checks: Ground passed at 5.00 s; Support
passed at 5.14 s, both with 51 sustained hand-contact frames. Event traces
confirmed accepted plans at exactly 0, 1, 2, 3, 4… simulation seconds,
actual student model and solverSteps=3. WebKit WebGPU Tangerine failed its
20 s one-Hz trial, so the cadence change is not a verified grasp improvement.

One-Hz WebKit WebGPU checks also failed Cup and Bottle at 20 s; all three
grasp objects failed this cadence trial. Pickup passes do not establish a
grasp improvement. The requested student/3-step/1-Hz default is retained.

Door side gaps: side jambs now extend to x=1.364, stopping 1 mm before the
closed panel front at x=1.365. Their inner edges are y=±0.410, overlapping
the closed panel silhouette by 15 mm on the hinge side and 9 mm at the latch.
Physical XML and JSON render sizes match. The panel, handle, bolt, and hinge
retain their dimensions/mass/limits. A native MuJoCo sweep of 121 hinge angles
from 0 to 3 rad found no frame contacts with the door assembly. A paused
Chromium screenshot verifies the closed view. This is clearance validation,
not a complete policy-driven door success test. Reload the local preview.

## Asynchronous RTC experiment (2026-09-28, superseded)

`rtc.js` implements the latency-aware deployment scheduler: snapshot state and
anchor at request time; infer in the existing worker without awaiting it in
the control loop; skip request-to-accept elapsed controls; SLERP the first five
body controls against any old remainder. Dex3 targets remain zero-order held,
with the same delay offset and no hand crossfade. Stale command/version/epoch
results and predictions older than their full horizon are rejected. One worker
request is allowed in flight. Goal edits can request an earlier update.

The actor remains FP32 but now defaults to CPU/WASM to avoid the planner GPU
queue blocking actuator updates. `?actor=webgpu` explicitly restores shared-GPU
execution for diagnosis. The planner stays the W8A32 distilled step-2000 student
at three steps, nominal one Hz. Both actor and planner/depth shaders are warmed
before Ready; warming does not step physics, consume sampler rows, mark a
command executed, or drop startup state. Teacher-only WebKit presets now also
require GPU actor execution, matching the profile in which they were measured.

Ordinary bank Walk runs directly as before. Empty buffers keep applying the
last latent and hand target; fallback does not mark a new command as executed.
With a one-second model horizon and one-Hz requests, some buffer underrun is
unavoidable during nonzero inference latency. This is continuous control, not
a guarantee of uninterrupted fresh predictions. Debug `rtc` counters expose
acceptances, expired plans, delay, buffered controls, and held controls.

Chromium and isolated WebKit tests both verified simulation advancement during
pending inference, first-plan state retention, invalidation on reset while a
request was in flight, and exactly one accepted state-drop plan after a genuine
command switch. Scheduler unit checks cover aligned body/hand delay, body
continuity without hand blending, stale version/epoch/command rejection, expiry,
and request cadence. After the first actuator step, observed update gaps were
p95/max 33.4/78.6 ms (Chromium) and 26.0/36.0 ms (WebKit). These tests ran
concurrently and are not isolated performance benchmarks. Actor 50 Hz remains
a target; each physical control advances simulation by 20 ms.

The six-second RTC trials had not yet reached sustained pickup success, so
prior synchronous success checks do not validate RTC task success. Reports:
`Artifacts/browser-reports/2026-09-28-rtc`. No commit, push, or publication.

## Synchronous student scene repair (2026-09-28, current)

RTC is disabled. Default planner is the W8A32 run-2 step-2000 student,
three Euler steps, one plan per 50 controls (1 Hz simulation time).
Inference pauses physics; actor remains FP32 and defaults to WebGPU when
available. The student planner defaults to WASM for the measured door/pickup
profile. Explicit `?backend=webgpu&actor=webgpu` remains available for diagnosis.
Do not compare wall-clock throughput with simulation-time planning cadence.

State is retained on the first simulation plan. A genuine executed command
change drops state on one accepted plan, except Carry while holding a box.
There is no automatic pause on success, falling, or reaching a waypoint.

Scene source is the local LSTracking checkout; its public GitHub URL returned 404. Dex3 uses per-clip support poses/sizes, not a universal support height.
All three resets use training rest arms (.2 shoulder pitch, +/-.2 roll, .6
elbow), open fingers, and one object. Exclude only each palm/thumb-1 convex
self-overlap; every object contact remains enabled. Planner finger targets
are applied directly, without a distance-dependent closure gate. Bottle and
cup mass use density 300 kg/m^3 and the equivalent box volume calculated
from convex collision inertia, clamped to .02–1 kg, as in training.

Bottle uses clip water_bottle_2__001: support top .76952 m, default root [.21,.65], WebKit root [.21,.72],
yaw -90 degrees, cold sampler 7. Tangerine support top .37443 m and Cup
.79 m are likewise recorded. Run restore_recorded_grasp.py followed by
repair_grasp_rest_pose.py on the export node to regenerate anchors.

The door reset [.15,-.6] positions the handle for the left hand. Door gets
no waypoint. After opening, a separate Walk bank stage crosses toward
[2.3,-.15] and aligns heading before a three-second physical stance settle.
The box is at [3.6,-.15], dimensions [.24,.32,.34], mass .5 kg. Pickup gets
no waypoint. Cold and Run-three-seconds-to-Chain trials both achieved the
strict one-second physical pickup gate. The later carry return is not yet
reliable: both trials dropped the box before completing the return.

Success checks require >.1 m lift, 51 raised control frames, >=45 hand
contact frames above .1 N, upright robot, and object within .8 m horizontally.
Reports are under Artifacts/browser-reports/2026-09-28-student-1hz-scene-tuning.
WebKit automation is a Safari-engine check, not a test of the user's actual
Safari installation. Fresh-start and command-switch trials are distinct;
see the result report for outstanding failures. No publication is authorized.

Bottle has measured engine-specific approach starts, seven centimeters apart.
The intermediate y=.685/.70 starts failed the Run-to-Grasp trial in both
engines. This is a scene profile, not different network precision or solver
steps. Cold sampler remains 7; switch sampler remains 3. Reset-time root XY
translation does not alter the BFM anchor's root-relative observation.

Reproduce local browser grasp checks with Playwright installed:

```sh
node tools/fiber-player/audit_student_grasp.cjs /tmp/fiber-grasp-cold-chromium
BROWSER=webkit node tools/fiber-player/audit_student_grasp.cjs /tmp/fiber-grasp-cold-webkit
WARM=1 node tools/fiber-player/audit_student_grasp.cjs /tmp/fiber-grasp-warm-chromium
BROWSER=webkit WARM=1 node tools/fiber-player/audit_student_grasp.cjs /tmp/fiber-grasp-warm-webkit
```

`FIBER_PREVIEW_URL` overrides localhost:8001/playground. Warm means Run
for about three simulation seconds, then Grasp without reset. These are
curated deterministic trials, not an estimate over randomized seeds.


## Carry waypoint restoration (2026-09-28)

At the user's request, restored the previous command() waypoint calculation:
carry displacement follows the waypoint bearing directly, capped at 1 m/s
over each horizon; heading follows the selected orientation with the existing
yaw control rate. Removed the added forward-arc clamp, turn-dependent speed
scaling, and chain-specific forced clockwise steering. Scene fixes, the student,
three steps, synchronous 1 Hz, and Walk's direct z-bank control remain current.
Earlier chain return reports describe the experimental arc calculation.


## Box contact export correction (2026-09-28)

Found native export requests box_condim=4 and torsional friction=.2, but
serialized MJCF omitted condim. Since the box has priority=1, it forced
three-dimensional contacts even against condim=4 hands; torsional resistance
was unused. repair_box_contact.py restores explicit condim=4 for all box
variants, including the door scene. Exporters now serialize the runtime
geom_condim. Sliding friction remains 1.2 and all other physics settings are
unchanged. Grasp objects already use condim=4 and hand contact sliding
friction 1.8; their parameters are unchanged. These approximate MuJoCo
contacts, not an exact PhysX material/patch-contact reproduction.
