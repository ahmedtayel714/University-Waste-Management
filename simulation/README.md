# Webots Simulation — Conceptual Demonstration

Scope, decisions, and the day-by-day plan live in
[`Simulation_Sprint_Plan.md`](Simulation_Sprint_Plan.md) in this same
folder. Short version: this
simulates the robot's *body* (a Webots library robot, not real hardware)
being driven by the project's *real* leaf detector. Any recording of this
must be captioned "Simulation — Conceptual Demonstration" — never
presented as a real robot.

## What's here

```
simulation/
  worlds/
    leaf_field.wbt          # the scene: arena + 5 scattered leaves + e-puck robot
    textures/leaf.jpg       # real leaf photo (data/field_validation/images/green_leaf_01.jpg), used as leaf texture
  controllers/
    leaf_collector/
      leaf_collector.py     # main controller: camera -> real model -> steer -> scripted pickup
      vision_bridge.py      # loads the real leaf model once, wraps it for per-frame use
```

**Status: verified against a real headless Webots R2025a run.** The world
file loads cleanly — E-puck + scoop attachment + 5 textured leaf Solids
all resolve with no PROTO/field errors, and the controller launches,
imports `controller` and `vision_bridge`, and reaches model loading. The
only thing stopping a full run is that no weights file is present yet
(step 3 below) — that's expected, not a bug.

## Setup

1. **Install Webots** from [cyberbotics.com/#download](https://www.cyberbotics.com/#download)
   directly — not the Homebrew cask, which is deprecated (fails Gatekeeper,
   gets disabled 2026-09-01). If macOS blocks the app on first launch:
   right-click → Open, or System Settings → Privacy & Security → "Open Anyway".
   (Already done and verified on this machine as of this commit.)

2. **Point Webots at a Python that has your project's dependencies.**
   The controller needs `ultralytics`, `opencv-python` (the GUI build, not
   `-headless` — the controller opens a live preview window), and `numpy`
   in whatever Python Webots launches the controller with. In Webots:
   Tools → Preferences → General → Python command, set it to the Python
   you want (e.g. the same one you're using for this project locally).
   Webots injects its own `controller` module into that interpreter's
   path automatically — you don't pip-install that part.

3. **Copy real trained weights next to the controller** (Drive-mounted
   paths won't be reachable from your local machine):
   ```bash
   cp /path/to/your/downloaded/leaf_v3/best.pt \
      simulation/controllers/leaf_collector/leaf_v3_best.pt
   ```
   Or set `LEAF_WEIGHTS_PATH` as an environment variable instead of
   copying the file, if you'd rather not duplicate it.

4. **Open `simulation/worlds/leaf_field.wbt` in Webots** and hit play.

## What to expect / first things to check

- Loading the world produces one harmless warning about the leaf texture
  not being a power-of-two size (Webots auto-rescales it) — that's fine,
  ignore it.
- The controller prints `[leaf_collector] ...` lines to Webots' console —
  a "loaded real leaf model from ..." line confirms the weights loaded;
  "collected leaf #N" lines confirm the pickup logic is firing.
- A second window ("leaf_v3 — robot camera (first-person)") shows the
  live annotated camera feed with detection boxes — this is the shot to
  screen-record for the first-person half of the dual-camera video. Screen
  -record Webots' main 3D viewport separately for the third-person half.
- `APPROACH_BOX_HEIGHT_FRAC` in `leaf_collector.py` controls how close the
  robot needs to get before triggering a pickup — tune this after seeing
  the first real run, since it depends on the e-puck camera's actual
  field of view at your chosen resolution.

## What this deliberately does not do

Real grasp planning, inverse kinematics, pixel-to-real-world camera
calibration, and matching a 2D detection to its exact 3D leaf node are
all out of scope (see the sprint plan §3). The scripted "pickup" instead
finds whichever remaining leaf is geometrically nearest the robot's own
(ground-truth, Supervisor-known) position — steering is driven by the
real model's real detections, but "which leaf did I just collect" is
resolved with information a real robot would need its own localization
stack to know, which is intentionally a separate, later problem.
