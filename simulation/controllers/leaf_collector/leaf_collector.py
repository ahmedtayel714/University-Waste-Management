"""Webots controller for the leaf-collection conceptual simulation.

Robot: e-puck (Webots library model — differential drive + built-in
camera), with a simple scoop shape attached in its extension slot — see
simulation/worlds/leaf_field.wbt. Per the sprint plan, this is
deliberately NOT a custom-CAD robot or an articulated arm.

Per-step pipeline:
  1. read the robot's camera frame (real pixels from the sim, not synthetic)
  2. run it through the REAL, already-trained leaf model
     (LeafVisionBridge — no retraining, no shortcut, same weights file
     used in the actual project)
  3. steer toward the highest-confidence detected leaf with simple
     proportional control on its horizontal offset from frame center —
     this part is genuine visual servoing driven by a real detection
  4. when close enough, run a scripted "pickup": stop, remove the
     nearest remaining leaf node from the scene via the Supervisor API

IMPORTANT — what step 4 is and is not: mapping a 2D detection box back
to a specific 3D leaf in the world requires camera calibration
(pixel -> real-world coordinates), which is explicitly out of scope for
this simulation (it's a real future roadmap item, not solved here). So
"which leaf did I just collect" is resolved using the robot's own
ground-truth 3D position via the Supervisor API (ok to use — a real
robot doing this would need its own localization to know it too, and
that's a separately-scoped problem). The STEERING, however, is driven
entirely by the real vision model's real 2D detections — that part is
not staged.

There is no real grasp planning here (also explicitly out of scope) —
"collecting" a leaf means removing its node from the scene, a staged
stand-in for a physical pickup.

Always caption any recording of this as "Simulation — Conceptual
Demonstration" — never as a real robot. See Simulation_Sprint_Plan.md
section 5 for the exact framing used with judges.
"""

import math
import os

import cv2
from controller import Supervisor

from vision_bridge import LeafVisionBridge, camera_image_to_bgr

# ---------------------------------------------------------------- config --
WEIGHTS_PATH = os.environ.get(
    "LEAF_WEIGHTS_PATH",
    os.path.join(os.path.dirname(__file__), "leaf_v3_best.pt"),
)
CONF_THRESHOLD = 0.25
MAX_WHEEL_SPEED = 6.28            # e-puck's max angular wheel speed (rad/s) — hardware limit, do not exceed
APPROACH_BOX_HEIGHT_FRAC = 0.55   # detected box height / frame height -> "close enough to collect"
TURN_GAIN = 4.0                   # proportional gain: horizontal offset -> wheel speed differential
FORWARD_SPEED = 3.0
SEARCH_TURN_SPEED = 1.0           # slow rotate-in-place when nothing is detected
LEAF_DEF_PREFIX = "LEAF_"         # world file must DEF-name leaves LEAF_1, LEAF_2, ... for Supervisor lookup
SHOW_FIRST_PERSON_WINDOW = True   # cv2.imshow of the annotated camera feed, for the dual-camera recording

# ------------------------------------------------------------- Webots setup --
robot = Supervisor()  # Supervisor, not plain Robot — needed to query/remove leaf nodes by DEF name
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

vision = LeafVisionBridge(WEIGHTS_PATH, conf=CONF_THRESHOLD)
print(f"[leaf_collector] loaded real leaf model from {WEIGHTS_PATH}")

# Discover every LEAF_n node placed in the world file up front.
leaf_nodes = {}
i = 1
while True:
    node = robot.getFromDef(f"{LEAF_DEF_PREFIX}{i}")
    if node is None:
        break
    leaf_nodes[f"{LEAF_DEF_PREFIX}{i}"] = node
    i += 1
if not leaf_nodes:
    print("[leaf_collector] WARNING: no LEAF_1, LEAF_2, ... DEF nodes found in the world — "
          "add scattered leaf Solids with those DEF names, see leaf_field.wbt.")
remaining_leaves = set(leaf_nodes)
leaves_collected = 0

self_node = robot.getSelf()  # the robot's own Node, for ground-truth position


def _flat_distance(pos_a, pos_b) -> float:
    """2D ground-plane distance (ignore height) between two [x, y, z] positions."""
    return math.hypot(pos_a[0] - pos_b[0], pos_a[2] - pos_b[2])


# ----------------------------------------------------------------- main loop --
while robot.step(timestep) != -1:
    frame = camera_image_to_bgr(camera)
    detections = vision.detect(frame)

    if SHOW_FIRST_PERSON_WINDOW:
        annotated = vision.annotate(frame, detections)
        cv2.imshow("leaf_v3 — robot camera (first-person)", annotated)
        cv2.waitKey(1)

    if not remaining_leaves:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        continue

    if not detections:
        # nothing detected this frame — rotate in place to search
        left_motor.setVelocity(-SEARCH_TURN_SPEED)
        right_motor.setVelocity(SEARCH_TURN_SPEED)
        continue

    target = max(detections, key=lambda d: d.conf)
    x1, y1, x2, y2 = target.box
    frame_h, frame_w = frame.shape[:2]
    box_cx = (x1 + x2) / 2
    box_h = y2 - y1
    offset = (box_cx - frame_w / 2) / (frame_w / 2)  # -1 (far left) .. 0 (centered) .. +1 (far right)

    if box_h / frame_h >= APPROACH_BOX_HEIGHT_FRAC:
        # Close enough — scripted "pickup" (see module docstring: not real
        # grasp planning). Resolve WHICH leaf via ground-truth 3D position,
        # since 2D-box-to-3D-node correspondence needs calibration we don't
        # have (and don't claim to have).
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        robot_pos = self_node.getPosition()
        nearest_def = min(
            remaining_leaves,
            key=lambda d: _flat_distance(robot_pos, leaf_nodes[d].getPosition()),
        )
        leaf_nodes[nearest_def].remove()
        remaining_leaves.discard(nearest_def)
        leaves_collected += 1
        print(f"[leaf_collector] collected leaf #{leaves_collected} "
              f"({nearest_def}, detection confidence={target.conf:.2f})")
        continue

    turn = TURN_GAIN * offset
    left_speed = max(min(FORWARD_SPEED + turn, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)
    right_speed = max(min(FORWARD_SPEED - turn, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
