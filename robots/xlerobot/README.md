# Direct XLeRobot control with Codex

[中文](README.zh-CN.md)

RPent runs Codex against a selected **hardware** or **simulation** backend.
Direct hardware operation does not start or require Isaac, Pi0.5 or PPO.
A registered network whole-body controller can provide actual state and camera
images; its autonomous readiness proofs remain authoritative. Configure the
explicit owner origin, profile SHA and camera sources in
[the network template](control.network.example.json). Codex
observes camera images, requests bounded tools, and checks their actual results;
the existing motor owner remains responsible for real-time execution.

## Direct hardware tasks

The location panel at `http://localhost:8899/#location` now has an explicit backend
selector. Missing `backend` in old API requests still means `simulation`.
A task keeps its backend until it ends. Readiness and individual skill blockers
are reported separately. Hardware views do not require model or mapping services.

The software path is 8899 UI → 8900 authenticated relay → 8919 RPent → 8920
calibrated skills → existing 8916 motor owner. The optional model endpoint remains
8095. The skill service must be explicitly configured, installed and commissioned;
source changes do not update the currently running services.

The Toolkit exposes `observe_robot`, `robot_status`, bounded base/native-joint
moves, `locate_object`, `navigate_to_workspace`, `grasp_object`, `verify_grasp`,
`task_status`, `stop_robot` and `finish`. Simulation also exposes USD joint moves
and diagnostic `pi05_predict`. Camera tools return image content, not only paths.

Managed planners on both backends run inside bubblewrap with a private task
workspace and task-scoped credentials. Linux Landlock restricts TCP access to the
RPent API, the task MCP port and the configured model port. No host robot
credentials or serial devices are mounted. Existing Codex login may be copied
into the private planner home; a configured endpoint/key file is optional.

`request_id` is single-use. Query `GET /v1/rpent/tasks/{id}` and `/events` after
an ambiguous response; do not submit another movement. `run_status`, `stage`,
`task_outcome` and `stop_state` are separate. An exit code of zero does not prove
pickup. Success requires trusted grasp evidence: at least 5 cm of lift, 3 seconds
of holding, visual following and gripper feedback. STOP revokes pending work,
contacts the motor owner independently of optional services and waits for fresh
stopped feedback. Unknown STOP blocks new work. Restart never resumes motion.

The initial scope is a registered empty pitcher, one right arm and registered
workspaces. Raw depth, measured camera/robot transforms, private object templates,
verified controller identity, measured kinematics/limits/collisions and an
explicitly resumed skill executor are prerequisites. Missing evidence blocks the
operation. Navigation additionally requires a commissioned map, localization and
obstacle input. RPent does not enable torque or manufacture calibration values.

See [the implementation and commissioning guide](DIRECT_HARDWARE.md) and
[the private configuration template](control.config.example.json). This change
has hardware-free regression tests; physical trials and the 18/20 acceptance
criterion have not been performed. The existing simulation mapping remains an
optional, mutually exclusive control mode, disabled by STOP and never restored
automatically. Do not run unmanaged planners with the service administrator token.

The following sections document the **optional simulation inference diagnostic**
and its historical results; they are not evidence of hardware grasp capability.

## Standalone inference diagnostic

The RPent checkout, `env_pi05`, `models/openpi`, and `xlerobot_isaac` must be
siblings. RPent has its own `.venv`; the model uses `env_pi05`; the existing
simulator launcher uses `env_isaaclab`. The adapter does not upgrade these
model or simulator environments.

From the parent workspace:

```bash
./start_rpent_xlerobot.sh
./start_rpent_xlerobot.sh status
./start_rpent_xlerobot.sh infer --prompt "Keep both arms still." --repeats 2
```

The default command starts or reuses the Isaac Sim GUI and the model server,
runs two live inference samples, and leaves both services running.
`start --no-verify` starts the services without inference.
`stop-model` shuts down only the model server, keeping Isaac Sim open.
The HTTP endpoint is bound to `127.0.0.1:8095`.

To invoke the diagnostic directly from the RPent checkout:

```bash
.venv/bin/python -m robots.xlerobot.diagnostic \
  --endpoint http://127.0.0.1:8095 \
  --output logs/xlerobot/my-verification \
  --prompt "Keep both arms still." --repeats 2
```

The output directory must be new. `--runtime` can select another local
`codex_runtime` directory. This does not configure physical hardware.

## Input and output contract

- Head, left-wrist and right-wrist RGB frames map to `base_0_rgb`,
  `left_wrist_0_rgb`, and `right_wrist_0_rgb`. OpenPI resizes them to 224×224.
- State order is left then right, each with Rotation, Pitch, Elbow,
  Wrist_Pitch, Wrist_Roll, and Jaw. Twelve absolute joint angles are mapped
  from the reconstructed USD joint limits to [-1, 1], tokenized with the
  instruction, and padded to the model's 32-dimensional state representation.
- This joint-limit normalization is a diagnostic convention, **not** training
  dataset statistics. Head and wheel joints are excluded.
- The full 3,353,433,872-parameter Pi0.5 model uses BF16 and ten flow-matching
  steps. Each result is a raw normalized `float32[50,32]` action tensor.
- Model outputs are saved and never sent to motors. There is no calibrated
  XLeRobot action mapping, XLeRobot fine-tuned checkpoint, or task-success claim.

The standalone diagnostic client stops the base before sampling and in its cleanup path. Its
simulation wrapper accepts only `status`, `observe`, and `stop`. The model
server has no simulation connection.

## Evidence

Each run writes `report.json` and one directory per sample containing all
three camera PNGs, source `observation.json`, `model_input.npz`, `actions.npy`,
and `result.json`. `server_seconds` includes device synchronization; the first
request includes JAX compilation. `rpc_seconds` includes transport overhead.
`PASS` means real observations produced finite model outputs. It does not
mean a physical manipulation or navigation task succeeded.

The initial local run is in `logs/xlerobot/verification-first/report.json`:
25.68 seconds for the first request including compilation, 0.574 seconds for
the second. These measurements used the RTX 3060 12GB alongside Isaac Sim.
Startup and client logs are under `logs/xlerobot/`; simulator logs remain
under the sibling `xlerobot_isaac/codex_runtime/` directory.

The GPU1 deployment was also verified through the 8899 panel: a real Codex
session called observe → Pi0.5 inference → observe → Head_Yaw movement →
observe → stop → finish. Inference took 0.378 seconds for a finite 50×32 tensor;
Head_Yaw increased by 0.03066 rad for a requested 0.03 rad change. The preceding
bounded base task moved about 0.0135 m and stopped. Both checks used simulator
feedback; they did not validate grasping. Shoulder drift was observed during the
head test. Evidence is under the local
`/home/ai/AI_dev/work_logs/rpent-control-20261011/` directory.

Offline checks:

```bash
.venv/bin/pytest tests/unit_tests/robots/xlerobot -q
.venv/bin/ruff check --preview robots/xlerobot tests/unit_tests/robots/xlerobot
```

Executing Pi0.5 predictions in a closed-loop task requires a compatible fine-tuned checkpoint,
dataset normalization, an explicit joint/action convention, and validation of
the reconstructed robot's collisions and camera calibration.
