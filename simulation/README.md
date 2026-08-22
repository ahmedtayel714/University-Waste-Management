# Webots Simulation — Conceptual Demonstration

Scope, decisions, and the day-by-day plan live in
[`Simulation_Sprint_Plan.md`](Simulation_Sprint_Plan.md) in this same
folder. That plan's original scope was a simple wheeled robot with a
scripted pickup (still present in `controllers/leaf_collector/` as a
working fallback). This folder's current primary version is a larger,
knowingly-approved scope increase: a real mobile-manipulator arm doing
an actual physics-based grasp. Any recording of either version must be
captioned "Simulation — Conceptual Demonstration" — never presented as
a real robot.

## What's here (current primary version)

```
simulation/
  worlds/
    leaf_field.wbt                     # KUKA youBot + SandyGround floor + 5 physics leaves + clutter
    protos/YoubotWithGripperSlot.proto # Cyberbotics' real Youbot.proto, forked to add ONE gripper attachment point
    textures/leaf.jpg                  # real leaf photo, used as leaf texture
  controllers/
    youbot_leaf_collector/
      youbot_leaf_collector.py    # main controller: state machine (search/approach/pick/lift/deposit/reset)
      youbot_control.py           # real youBot arm IK + gripper + mecanum base, ported from Cyberbotics' own C library
      depth_projection.py         # real pinhole camera + depth back-projection math
      vision_bridge.py            # loads the real leaf model once, wraps it for per-frame use
    leaf_collector/                # earlier, simpler e-puck version -- kept as a fallback, not currently wired to leaf_field.wbt
```

## What's real vs. staged in this version

**Real:** detection (whatever `leaf_v3`/`leaf_v3_ft` actually outputs —
confidence scores are never hardcoded, anywhere), the pixel+depth→3D
coordinate transform (genuine pinhole projection using the camera's
actual intrinsics and the RangeFinder's actual depth reading), the arm's
inverse kinematics (`youbot_control.py` is a line-for-line port of
Cyberbotics' own `arm_ik()` for this exact robot, not reinvented), the
grasp (a real Webots `Connector` lock between gripper and leaf, with
real physics/collision on the leaf — not a scripted disappearance), and
the base's mecanum-wheel steering.

**Staged, and disclosed as such in code comments:** the base stops and
hands off to the arm once a leaf is within reach, rather than solving
base+arm motion jointly; "deposit" moves to a fixed pose over an onboard
bin rather than driving to a separate location. Both are scope choices
on parts that were never claimed as more than staged, not shortcuts on
the parts described as real above.

## Verification status — please read before assuming it just works

- **Controller logic: fully verified**, offline, against a mock Webots
  harness exercising the complete state machine (search → approach →
  pick → lift → deposit → reset, twice through). No crashes, all joint
  angles finite, depth projection math checked against a hand-derived
  geometric test case.
- **Arm IK: verified for producing finite, in-range solutions**, and for
  correctly rejecting out-of-reach targets rather than emitting garbage
  angles. **Not verified live.** One specific thing to check on your
  first real run: whether the arm reaches toward the correct side
  (left/right) of the actual detected leaf. If it reaches to the
  mirror-opposite side, that's a single sign flip — see the comment
  above `lateral_right = -left` in `youbot_leaf_collector.py`
  (`body_to_arm_frame`), which documents exactly why that sign was
  chosen and what evidence would say it's backwards.
- **The world file: could not be verified with the same headless
  Webots test used successfully for the earlier e-puck version.**
  Loading youBot via `webots --batch --no-rendering --mode=fast`
  hangs indefinitely with zero console output — and this reproduces
  identically with Cyberbotics' own *unmodified* `Youbot.proto`, not
  just this project's fork, so it isn't something introduced here. It
  may be specific to this robot's heavier mesh assets under that
  particular flag combination. **Please open `leaf_field.wbt` normally
  in the Webots GUI** (not headless) for the actual first test — that's
  also how you'll be running it for real, so this isn't a gap that
  matters beyond today.
- The `gripperSlot` fork was verified structurally (brace/bracket
  balance, correct field wiring) but not load-tested for the same
  reason above.

## Setup

1. Webots is already installed and Full Disk Access is already granted
   on this machine (done earlier in this project).

2. **Copy real trained weights into the new controller folder**:
   ```bash
   cp /path/to/your/downloaded/leaf_v3/best.pt \
      simulation/controllers/youbot_leaf_collector/leaf_v3_best.pt
   ```
   Or set the `LEAF_WEIGHTS_PATH` environment variable instead.

3. **Open `simulation/worlds/leaf_field.wbt` in Webots** (via the GUI —
   see the verification note above) and press play.

4. Webots' Preferences → Python command should already be set to
   `/opt/anaconda3/bin/python3` from the earlier e-puck setup — the new
   controller needs the same packages (`ultralytics`, `opencv-python`,
   `numpy`), already present there.

## What to expect / debugging order if something's wrong

Debug in this order, since later stages depend on earlier ones working:

1. Console shows `[youbot_leaf_collector] loaded real leaf model from ...`
   — confirms the controller started and found the weights file.
2. A window titled "leaf_v3 — youBot overhead camera" shows a live,
   annotated view from the robot's overhead camera. If detections never
   appear even on an obvious leaf, check the camera is actually pointed
   at the floor (the mount's pitch — `rotation 0 1 0 1.30` on the Camera
   node — is a first guess, not a measured value; adjust the angle if
   the view is mostly sky/walls or mostly the robot's own chassis).
3. Console prints `tracking leaf, confidence=..., depth=...` while the
   base drives toward a detected leaf — confirms the base's visual
   servoing and the RangeFinder are both working.
4. Console prints `real depth-projected target: body=(...) arm=(...)`
   followed by the arm moving — this is the moment to watch for the
   left/right sign issue mentioned above.
5. `collected leaf #N (real Connector lock, real physics)` — confirms a
   genuine grasp. If instead you see "gripper closed but no Connector
   presence — missed grasp", the arm reached roughly the right place but
   not close enough for the Connector's own proximity tolerance — small
   position or `APPROACH_STOP_DEPTH` tuning territory, not a logic bug.

## What this deliberately does not fabricate

Confidence scores are never hardcoded — whatever the model actually
outputs on the actual camera frame is what's shown and printed, always.
This is the one line that was held regardless of how much the rest of
the scope grew.
