# FIBER browser playground

Live FIBER ONNX inference and MuJoCo physics, inspired by ULTRA's
click-to-goal player. Changes are local; models are excluded from Git.

Serve this checkout with `python3 tools/fiber-player/serve_preview.py --port 8001`, then
open `http://localhost:8001/playground/`. Models load only after Launch.

## Current default (2026-09-29)

Run-2 step-2000 distilled student (`v27-distill2000`), W8A32 planner,
three Euler steps, synchronous 1 Hz of simulation time, FP32 actor.
RTC is disabled. The preview server enables COOP/COEP so the CPU planner
can use four WASM threads. Hosts without these headers retain one thread.
Reload the top-level page after switching servers to enable isolation.

Measured local medians: Chromium 1.54 s → 0.55 s, WebKit 1.47 s → 0.47 s
for one versus four threads. Repeated evaluation of the same captured door
input produced identical body and hand tensors in each engine. These are
local timing and parity checks, not cross-device performance guarantees.

Current deployment details are in `tools/fiber-player/ON_NODE_HANDOFF.md`.
The sections below record earlier export and tuning work, including teacher
and 2 Hz configurations that are no longer the default.

## Checkpoint and browser models

The player uses the latest evaluated **v27, step 226000**, with its v5 cached
language vectors. Checkpoint SHA256:
`181a5e9d2c3fdad68c1192ef4ca70174d3b49bd16777c820a978f24aafef2c7d`.
Tabletop grasp uses its dedicated `grasp_table` instruction; door opening
uses `open_door`.
The frozen BFM controller uses 256-dimensional executed latents.

Training checkpoints stay on the GPU node. Only compact inference artifacts
and scene assets are copied locally:

| Model                                       |     Size |
| ------------------------------------------- | -------: |
| Controller, FP32                            | 135.3 MB |
| Depth/condition encoder and attention K/V   | 100.2 MB |
| Cached timestep modulation                  | 174.5 MB |
| Repeated trajectory step, body + hand heads | 267.2 MB |
| Total                                       | 677.2 MB |

`models/v27/planner_meta.json` stores the trained state and Dex3 normalization
contracts. Fixed-shape FP32 ONNX was checked against PyTorch before
quantization. Final W8 sampled relative output errors were 1.59% for the body
and 1.36% for the hand head. These are numerical checks, not success metrics.

## Speed optimization

Interactions, carrying and precision turns keep eight Euler solver steps. Manual
walking uses four steps from the same cached timestep tensors. Depth features and every
cross-attention key/value are computed once per observation, then reused
as GPU tensors. The eight fixed timestep modulation tensors are computed
once at startup. All solver steps and their Euler updates stay on the GPU inside one worker
request. Only the final plan is read back, avoiding repeated copies, GPU
synchronization and messages. Rendering skips unchanged frames.

Graph splitting was checked against the unsplit compact v27 model at
multiple timesteps with command conditioning enabled and disabled: maximum
output difference **0.0** in the CPU checks, for both body and hand outputs.
The renderer and planner remain on separate threads.

## Remote export

Scripts are in `../tools/fiber-player/`. Run on the export node, with the
FIBER source available through `PYTHONPATH`:

Set `FIBER_MODEL=v27` for the planner export commands.

1. `export_planner.py`: checkpoint, FP32 parity, hand contracts, initial W8.
2. `compact_gemm.py`: compact remaining linear layers and check final W8 parity.
3. `split_planner.py`: condition/time/step graphs and split parity checks.
4. `fuse_sampler.py`: GPU Euler integration and full eight-step parity checks.
5. `export_grasp_scene.py`: articulated Dex3 scene and arrival metadata.
6. `export_pickup_scene.py`: held-out pickup arrival pose, support and geometry.
7. `refine_grasp_scene.py`: original tabletop mesh and pre-contact pose.
8. `export_demo_noise.py`: reproducible Gaussian sampler inputs.

The legacy `export_browser.py` prepares the controller and pickup/door assets.
Models should be published separately only after website approval.

## Scene and execution limits

Three.js 0.151.3 renders orbit and FPV cameras. The official `@mujoco/mujoco` 3.14.0 runtime runs physics, matching the native
reference runner.
ONNX Runtime Web 1.30.0 uses WebGPU, with WASM for compatible operations.
Planning replans at 2 Hz of simulation time with eight Euler steps, 50 Hz
control, multi-point spherical interpolation, and a five-control RTC blend.
Actual speed depends on browser and hardware.

The pickup and door scenes use welded closed Dex3 fists, matching the
successful MuJoCo chain runner. The grasp scene has articulated Dex3 hands. The **Tabletop grasp** scene
has all fourteen articulated Dex3 joints, driven by the v27 hand head in the
canonical joint order. Its hand targets use the checkpoint's joint-limit
normalization and are held over each knot's controls. PD gains and effort
limits follow the GRAIL robot configuration. The start is a pre-contact frame (275) of held-out GRAIL
`pickup_table__tangerine_4__002`. The fruit uses the original USD vertices,
its recorded orientation, MuJoCo convex collision, and density-derived mass
(0.168 kg). Reset velocities are zeroed.

The skill arbiter is explicit demo logic, not a trained model. This single
MuJoCo scene is not the batched paper evaluation. Success is experimental;
the UI labels falls, timeouts, and lifted-object holds. The v27 task controls
use the instruction modes trained in this checkpoint.

Pickup uses held-out arrival pose `ep_0004`, the matching box dimensions and
support, and a 0.5 kg box. XML exports preserve runtime geometry and mass
changes. Door uses the handle standoff (0.58, -0.33), faces the lever, and
retains its physical latch.

Demo sampler seeds are fixed (pickup/door 42; tangerine 15, water bottle 4, cup 52). The small noise
files contain only Gaussian inputs to inference, not actions, plans, or
reference trajectories. The 100 input rows repeat after 50 simulation
seconds. Reset repeats the same inputs, making regressions reproducible.

Lift completion requires a rise over 0.1 m for 51 control samples and
physical hand contact over 0.1 N on at least 45 of those samples. Door
completion requires hinge angle over 0.5 rad and root x beyond 1.8 m.
The player pauses at completion. These are demo checks, not paper metrics.

## Attribution

UI interaction pattern: https://ultra-humanoid.github.io/playground/
MuJoCo: https://github.com/google-deepmind/mujoco (Apache 2.0)
Three.js: https://github.com/mrdoob/three.js (MIT)
ONNX Runtime: https://github.com/microsoft/onnxruntime (MIT)
Copies of runtime licenses are in `vendor/`.
The scenes and policies originate from the FIBER project.

## Alignment and controls audit

Reset history follows the native runner: zeroed lag slots, current reset frame
in slot zero. Live depth comparisons at two matching poses had mean errors
of 1.7 / 3.9 mm, with some larger rasterization differences at moving edges.
Replaying browser actions in Python MuJoCo gave a maximum pose difference
of 9.1e-9 after thirty control steps. Eight-step compact planner samples
were within 0.8% relative error of PyTorch on these inputs.

The controller uses FP32 ONNX: INT8 produced 2.7–6.5% relative action errors
on sampled live inputs, and FP16 produced non-finite outputs on WebGPU.
WebGPU FP32 controller actions matched CPU ONNX within 1.4e-6 on 25 live inputs.
Only the inference graph is transferred; training checkpoints stay remote.

Q / E command separate left / right 30-degree headings with zero commanded
translation. W / S / A / D command translation while retaining the heading.
White arrows show current root yaw. Blue arrows show the actual conditioned
positions and yaw at the two planner horizons, frozen at each planning pose.
Guides are excluded from FPV and depth. The status card displays the cached
instruction text. Interaction prompts with dropped waypoint conditioning
do not display conditioned guides.

A full-precision native pickup rollout from the original standing demo scene
failed to lift the box through sixteen simulation seconds. Initial depth
alignment alone does not establish pickup success.

Browser turn smoke tests reached the ±30° targets within the 0.08-rad
completion tolerance at 2.64 / 3.98 seconds of simulation time.

## Historical fixed-start rollouts (2026-09-27)

Apple Metal WebGPU, compact v27 planner, FP32 controller, MuJoCo 3.14.0:

| Task       | Completion, simulation seconds | Physical evidence                                          |
| ---------- | -----------------------------: | ---------------------------------------------------------- |
| Pickup     |                           2.26 | peak rise 0.404 m; 51 raised samples; hand contact 48/51   |
| Door       |                           9.54 | hinge 1.345 rad; root beyond the door at x=1.805 m         |
| Dex3 grasp |                           3.54 | peak rise 0.352 m; 51 raised samples; finger contact 51/51 |

Planning took 0.866–0.873 seconds in these tests. These validate the fixed
demo starts and sampler seeds, not a general success rate or arbitrary goals.
The experimental door–pickup–return chain is not offered in the public UI;
these checks cover standalone tasks.

The browser previously used MuJoCo 3.3.8 while the native runner used 3.14.0.
Grasp outcomes changed when using the older native engine too. Matching
engine versions fixed the tested browser grasp without a policy replacement.
Runtime API reference: https://github.com/google-deepmind/mujoco/blob/main/wasm/README.md

## Current controls and layouts

- Walk: left-click the floor to place a goal; drag from that point to choose
  the heading. Right-drag orbits the camera. The X/Y fields and heading
  slider update the goal while moving, without resetting physics.
- Q/E: tap to change heading by 30 degrees; hold to continue turning.
  The command yaw cap defaults to 1.8 rad/s and is adjustable up to 3.
  This is the conditioning cap, not a guaranteed physical turning speed.
- WASD adjusts the walking goal. Release keys to keep the latest goal;
  Pause stops execution; Reset restores the chosen layout.
- Select a box, choose Pickup, then select Carry to place its destination. Carry uses the same
  click/drag, heading slider, coordinate fields and keyboard controls as Walk.
  Carry is enabled only after a confirmed lift; Walk also keeps the carry
  policy when a box is held.
  Mixed, all-ground and all-support layouts each contain three physical boxes.
  Selecting a different object preserves the robot's position and pauses
  the current task. Distant targets get a walking approach before interaction.
- Grasp offers separate tangerine, water bottle and cup scenes. Object selection loads
  a fresh simulation with exactly one tabletop object, matching the single-object
  setup used to train the planner. Each scene starts with relaxed arms down and
  all 14 finger joints open. These meshes come from the existing grasp dataset; the interactive defaults
  are separate from reported evaluation results. Switching objects resets the scene and pauses execution.

Live goal edits are smoothed over 0.14 seconds and coalesced between plans.
Edited goals trigger a new plan after at least five executed control steps;
execution anchors always use the last actually executed latent. Remaining
controls blend into the next plan. A running inference request is allowed
to finish, so new commands still have inference latency.

Current Apple Metal browser checks:

| Case                                     | Result                                                                         |
| ---------------------------------------- | ------------------------------------------------------------------------------ |
| Mixed layout, first supported box        | lift 0.404 m; held/contact 51/51; completes at 2.28 s                          |
| Ground layout, first box                 | lift 0.689 m; held/contact 51/51; completes at 2.10 s                          |
| Door regression                          | hinge 1.345 rad; root x=1.805 m; completes at 9.54 s                           |
| Walk click/drag + live X/Y/heading edits | updates during execution without reset; four-step plans about 0.52–0.54 s      |
| Carry click/drag + live edits            | carry conditioning preserved; contact 51/51 and 104 raised samples after edits |
| Neutral grasp initialization             | relaxed resting arms and 14 open finger joints verified                        |

Each grasp scene starts with lowered arms and all 14 fingers open. The selectable objects are Tangerine, Water bottle and Cup, each in its own scene. Default sampler seeds are now 15, 4 and 52 respectively; no initial state token is dropped.

Pickup selection loads a fresh single-box scene. All nine ground/support placement choices are retained, with low support using the original held-out episode 14 arrival, box dimensions and support geometry. Run `export_pickup_single_cases.py`, then `export_pickup_low_candidates.py` to reproduce the assets. No object forces, live pose overrides or altered success thresholds are used.

## Command transition checks

State dropout is triggered by a change from the last actually executed model
command, for one accepted plan. Starting a simulation or restarting after reset
retains state on the first plan. Repeating a command or editing its waypoint
does not drop state. The following plan restores conditioning automatically.
Actor proprioception is always retained. Carrying a held object retains planner
state conditioning. Phase/status changes alone do not trigger a drop.

Object selectors now use illustrated tiles and layout tabs. The selected
object uses an underline with `aria-pressed`; the legacy native layout select
is hidden and retained as the event interface.

### Box state-conditioning ablation

Native FP32 v27, mixed layout first supported box, seed 42: dropping the
planner state token for the first pickup plan failed the physical hold gate
(max lift 0.173 m, best raised hold 20 samples). Retaining state passed at
2.28 sim seconds (max lift 0.397 m, hold 51 samples, contact 48/51).
The first pickup plan now retains state. Subsequent command switches follow
the transition rule above.
The full selectable browser defaults are audited separately below.

## Browser default audit

`tools/fiber-player/audit_browser_defaults.cjs` drives the actual scene/layout/object selectors and task buttons in Chromium with Metal WebGPU. Install Playwright in your Node environment and run:

```sh
node tools/fiber-player/audit_browser_defaults.cjs /tmp/fiber-browser-audit/defaults
```

The output includes JSONL state/contact evidence and a screenshot per trial. Set `FIBER_PREVIEW_URL` to change the local preview URL, or `CASES=pickup-support-2,grasp-3` to test specific defaults.

Success uses the actual UI completion gate: a box/object raised over 0.1 m, held for 51 consecutive control samples with at least 45/51 hand-contact samples, or the door opened and crossed. These fixed-seed default-case checks are not a randomized policy success-rate estimate. The full door–pickup–return chain remains experimental.

Original multi-box browser baseline: 5/13 (pickup 3/9, door 1/1, grasp 1/3). The fresh-scene trials and sampler defaults address failures without relaxing the physical contact gate.

Current optimized Chromium Metal defaults: **13/13 passed** (pickup 9/9, door 1/1, grasp 3/3). Box completion times: 2.10–2.40 sim seconds. Grasp: tangerine 8.24s, water bottle 12.46s, cup 9.92s. Door: 10.80s. Cup seed 9 passed native FP32 but failed browser W8; seeds 21 and 37 also failed browser, including a 5 cm lateral-placement test. Final cup uses seed 52 at its original tabletop position. The first plan retained state in every default; the door command transition dropped exactly one plan.

See `tools/fiber-player/browser-default-results.json` for per-case contact, hold, state-drop and timing evidence, plus the original baseline. This is one fixed-sampler trial per configuration in the final set, not a population success-rate claim.

### Safari/WebKit correction

The 13/13 count above is specific to automated Chromium Metal trials, not all browser engines. WebKit 26.6 on the same Mac reproduced direct-start grasp failures for all three objects (0/3). A short 0.5-second standing run before Grasp also failed 0/3; the user reports bottle and cup success after a longer standing run in Safari. Startup retains planner state; an actual walk-to-grasp command switch drops it for exactly one plan and supplies accumulated controller history. The duration also advances the sampler stream, so these are different conditions.

For a direct Water bottle start, both engines produced identical depth, normalized state and initial Gaussian inputs. First planner output maximum differences: body 3.58e-6, hand 1.31e-6. First actor output difference: 6.74e-7. Physics diverges during the first control step and compounds afterwards. This does not establish the source of the divergence or prove that browser support is fixed.

Planner file sizes: condition 100,216,962 bytes; time 174,468,086; step 267,158,773; total 541,843,821 (517 MiB). Actor: 135,281,319 bytes. The source FP32 ONNX holds 478,068,750 float initializer entries and is 1,913,528,147 bytes. The loader now labels MiB correctly and displays cumulative bytes. Loaded checkpoint/precision/backend/file sizes are also available in `fiberDebug.state`.

A 3-second standing run also failed 0/3 in isolated WebKit 26.6, while the user reports bottle/cup success in native Safari 26.5. Disabling robot self-collision as a staged candidate did not fix the 0/3 direct-start grasp failures, so no collision-setting change was promoted. Browser timing and runtime differences remain unresolved.

Controls → Save run report downloads a local JSON file with command/plan timing, actual state-drop events, sampler rows, loaded model metadata and final simulation state. It does not upload data. This is intended to capture actual Safari trials when native app access is denied by computer-use permissions.

`tools/fiber-player/capture_webkit_reports.cjs` automates Reset → Run until 3 seconds of simulation → Grasp directly while running, then downloads the actual Save run report file for each object. The reports identify isolated WebKit 26.6 and must not be presented as captures from the user's native Safari 26.5 tab. Native Safari access was denied again after the user authorized capture.

## Distilled planner trial (run 2, step 2000)

The local preview defaults to the supplied `v27-distill2000` W8 split export at **3 Euler steps**. Its metadata identifies `v27-distill2000-228000`, SHA256 `d8441d6cdaf24d01e7a0ceb9fd309c31c5e42881fb55f0d193504986df56a483`. State, hand, language, waypoint and actor contracts match the original v27 export exactly. The actor remains FP32. Metadata relative quantization errors are 1.38% body and 1.05% hand.

The worker caches separate timestep schedules for 3, 4 and 8 steps. Student inference evaluates times 0, 1/3, 2/3; it does not reuse indices from the teacher's eight-step table. The fused Euler update uses `remaining_steps = 3, 2, 1`. Commands, startup state retention and one-plan dropout on actual command switches are unchanged.

Use `/playground/?planner=teacher` to compare against the retained original v27 teacher (8 interaction steps, 4 manual walking steps). Add `backend=wasm` for a CPU-path measurement. Default inference uses WebGPU when available. The source checkpoint and full FP32 export remain remote; only the three compact browser models and metadata were transferred, totaling 541,843,821 bytes of planner models.

The supplied on-node evaluation recommends this student at 3 steps. Its pickup outcomes (7/27 against teacher 10/27) do not establish equal success rates; those are a different protocol from the fixed browser UI defaults. Local default UI trials: Chromium Metal **10/13** (pickup 7/9, door 1/1, grasp 2/3); isolated WebKit 26.6 **3/5** (low-support pickup and door passed, tangerine passed, bottle/cup grasp failed). The failed Chromium pickup selections were the first medium supported box in Mixed and Support, and bottle grasp. One fixed-sampler trial per case is not a randomized success-rate estimate. These trials use separate one-object scenes and are not the supplied on-node 27-trial protocol. See `tools/fiber-player/distilled-browser-results.json` for actual model/solver/runtime and per-case contact evidence.

Isolated Chrome planner benchmark on this Mac, median of five warmed plans after two warmup samples, same fixed inputs:

| Backend                        | Teacher 8 steps | Student 3 steps | Speedup |
| ------------------------------ | --------------- | --------------- | ------- |
| Metal WebGPU                   | 0.831 s         | 0.391 s         | 2.13×   |
| Single-thread browser WASM CPU | 2.886 s         | 1.415 s         | 2.04×   |

These are full planner calls, including the scene encoder. Reducing Euler evaluations to 3/8 reduces the iterative work to 37.5%; fixed encoding overhead means measured full-plan latency remains about 47–49% of the teacher. The supplied 0.45 s CPU estimate was not reproduced with this single-thread browser configuration. All measured body and hand outputs were finite. `tools/fiber-player/benchmark_distilled.cjs` reproduces the comparison; set `BACKEND=wasm` for the CPU path.
