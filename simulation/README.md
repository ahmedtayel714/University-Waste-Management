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
  leaf/                               # 10 source leaf photos (plain white background) provided for texturing
  worlds/
    leaf_field.wbt                     # KUKA youBot + SandyGround floor + 9 physics leaves + clutter
    protos/YoubotWithGripperSlot.proto # Cyberbotics' real Youbot.proto, forked to add ONE gripper attachment point
    textures/leaves/*.png              # 9 alpha-cutout PNGs generated from simulation/leaf/ -- see below
    textures/leaf.jpg                  # earlier single-photo texture, no longer referenced (kept, harmless)
  controllers/
    youbot_leaf_collector/
      youbot_leaf_collector.py    # main controller: state machine (search/approach/pick/lift/deposit/reset)
      youbot_control.py           # real youBot arm IK + gripper + mecanum base, ported from Cyberbotics' own C library
      depth_projection.py         # real pinhole camera + depth back-projection math
      vision_bridge.py            # loads the real leaf model once, wraps it for per-frame use
    leaf_collector/                # earlier, simpler e-puck version -- kept as a fallback, not currently wired to leaf_field.wbt
```

### Leaf texture pipeline (why it changed)

The first version mapped a single real photo — `data/field_validation/images/green_leaf_01.jpg`, a full ground photo with one leaf on it, not an isolated cutout — onto a plain rectangle. Every leaf prop in the scene looked like a tiny framed photograph (dirt background and all) sitting on the ground, not like a leaf. Confirmed from a recorded run.

Fix: the 10 photos in `simulation/leaf/` (plain white background, one per file) were each run through `auto_segment_plain_background()` — the project's existing GrabCut-based cutout tool (`src/synthetic/cutout_extractor.py`, already used for Track B's synthetic dataset) — to produce real alpha-transparent PNG cutouts in `simulation/worlds/textures/leaves/`. One of the ten (`00009.jpeg`) turned out to be a decorative clip-art pattern of many tiny leaves, not a single-leaf photo, and was excluded; the other 9 are used, each on its own `Plane` sized to its image's real aspect ratio and a randomized target size (5–11 cm long edge). Alpha transparency means the ground actually shows around each leaf's silhouette instead of a background patch.

To regenerate or add more: see the one-off script logic in this project's history (`git log -p` on this README's introducing commit), or reuse `auto_segment_plain_background(image_bgr)` directly — it returns a 0/255 foreground mask from a plain/roughly-uniform-background photo.

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

Live-verified via actual recordings so far, each fixed in turn:
- World file loads and runs (youBot, arm, gripper, camera all functional).
- Camera self-occlusion by the arm's own mesh — fixed by raising/steepening the mount twice.
- False "leaf" detections on the arm's orange plastic at close range — fixed by gating detection to SEARCH/APPROACH only, raising the confidence threshold, and the camera geometry fix above; confirmed from a recording that the remaining detection was correctly on a real leaf-shaped object, not the arm.
- Ground/leaves rendering almost black — fixed a `DirectionalLight` pointing slightly *up* instead of down.
- Leaf props looking like small framed photos instead of leaves — fixed by switching to real alpha-cutout PNGs (see above).
- Robot stuck in APPROACH indefinitely with multiple leaves confidently detected but no net progress toward any of them — fixed a target-selection bug (`_select_target` was re-picking the single highest-confidence detection every frame; with several leaves visible at once, confidence-rank flips between them frame to frame caused the steering target, and turn direction, to keep switching). Now sticks with whichever detection is nearest the currently-tracked leaf's pixel position. Verified in isolation (both the "stays locked despite a rank flip" and "re-acquires when actually lost" paths) since a full-loop mock can't distinguish sticky-by-design from coincidentally-the-same.

**Known secondary issue, not yet dedicated a fix:** the paper-scrap and twig clutter props have occasionally drawn a low-confidence ("leaf 0.4–0.7"ish) false positive in recordings, well below genuine leaf confidences (0.8+). The sticky-target fix should mostly prevent this from ever being *acted on* once a real leaf is already locked, but if the very first detection after a SEARCH happens to be one of these, the base could still steer toward clutter briefly. Watch for this if leaves are ever momentarily out of frame.

**Not yet live-verified — arm IK left/right sign.** The math produces finite, in-range joint solutions and cleanly rejects unreachable targets (unit-tested), but whether the arm reaches toward the *correct* side of a real detected leaf hasn't been confirmed from a recording yet. If it reaches to the mirror-opposite side, that's a single sign flip — see the comment above `lateral_right = -left` in `youbot_leaf_collector.py` (`body_to_arm_frame`).

**Still not headlessly load-testable.** `webots --batch --no-rendering --mode=fast` hangs indefinitely loading youBot specifically (reproduces with Cyberbotics' own unmodified `Youbot.proto` too, not this fork) — every check above came from an actual GUI recording, which remains the only reliable way to verify a change here.

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
   annotated view from the robot's overhead camera, with the current
   state name printed top-left (useful for matching a recording to what
   the controller was doing at that moment). If detections never appear
   even on an obvious leaf, check the camera is actually pointed at the
   floor and not into the robot's own arm (the mount — `rotation 0 1 0
   1.35` on the Camera node, near-overhead — has already been corrected
   twice from recordings; if it's wrong again, this is the thing to
   adjust, keeping the Python controller's `CAMERA_MOUNT_TRANSLATION`/
   `CAMERA_MOUNT_ROTATION_AXIS_ANGLE` in sync with whatever you change).
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
