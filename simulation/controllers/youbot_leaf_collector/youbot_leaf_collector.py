"""Webots controller: youBot mobile manipulator, driven by the real,
already-trained leaf detector, doing an actual physics-based pick-and-
place of detected leaves.

What is real:
  - Detection: leaf_v3 (or whatever LEAF_WEIGHTS_PATH points at) runs on
    the actual camera frame every step. Confidence scores shown/printed
    are exactly what the model outputs -- nothing here hardcodes or
    adjusts them.
  - Coordinate transform: pixel + RangeFinder depth -> 3D position uses
    real pinhole-camera back-projection (depth_projection.py) with the
    sensor's actual mounted geometry -- not a fabricated number.
  - Arm kinematics: youbot_control.YoubotArm is a direct port of
    Cyberbotics' own arm_ik() for this exact robot (see that file).
  - Grasp: a real Webots Connector lock between the gripper and the leaf
    (see samples/devices/worlds/connector.wbt for the reference pattern)
    plus real gripper finger closure -- the leaf has real physics and
    collision, it is not removed/teleported.
  - Base motion: real mecanum-wheel control, real proportional visual
    servoing on the detection's pixel offset.

What is staged, and why, documented at the point it happens below:
  - The base stops and hands off to the arm once a leaf is within the
    arm's reach envelope, rather than solving base+arm motion jointly
    (a real mobile-manipulation system could do either; this picks the
    simpler, still-genuine option deliberately).
  - "Deposit" moves the grasped leaf to a fixed pose above an onboard
    bin rather than driving the base to a separate collection point --
    scope choice, not a shortcut on the parts that are actually real.

Always caption recordings "Simulation -- Conceptual Demonstration".
"""

import math
import os

import cv2
from controller import Robot

from depth_projection import RgbdProjector
from vision_bridge import LeafVisionBridge, camera_image_to_bgr
from youbot_control import YoubotArm, YoubotBase, YoubotGripper

# ---------------------------------------------------------------- config --
WEIGHTS_PATH = os.environ.get(
    "LEAF_WEIGHTS_PATH",
    os.path.join(os.path.dirname(__file__), "leaf_v3_best.pt"),
)
# Raised from 0.25 -- a recorded run showed the arm's own orange plastic
# (never seen in training; plausibly close in hue to the dry-leaf tint
# augmentation) drawing 0.2-0.5 confidence false positives at close
# range. This does not fix the root cause (still visible in whatever
# survives at higher confidence) but cuts the weakest, most obviously
# wrong hits. See simulation/README.md.
CONF_THRESHOLD = 0.45
CAMERA_WIDTH, CAMERA_HEIGHT = 256, 256
CAMERA_FOV = 0.9  # radians -- MUST match the Camera node's fieldOfView in leaf_field.wbt

# Overhead RGB-D mount, in the robot BODY's local frame (x=forward,
# y=left, z=up). MUST match the Camera/RangeFinder translation+rotation
# in leaf_field.wbt exactly -- these numbers are not independent of the
# world file, they describe the same physical mount.
#
# Raised from an earlier low mount (z=0.30) that put the camera inside
# the arm's own operating envelope -- confirmed from a recorded run that
# the camera was staring at the arm's own dark mesh, not the ground.
# That fix (z=0.55, ~57 degrees) was NOT enough -- a recorded run during
# an active state still showed the arm filling most of the frame.
# Went higher and steeper (closer to straight down) instead of just
# higher: a near-overhead view has much less grazing-angle parallax, so
# a foreground object of a given height occludes less of the frame than
# the same object seen from a shallow angle. This narrows the visible
# ground patch to roughly x=0.15-0.35m ahead of the mast (verified via
# the same projection math used at runtime) -- still geometrically
# overlapping where the arm CAN reach, since the camera has to see the
# ground the arm operates on. Detection is now also paused outside
# SEARCH/APPROACH (see step() below) so an extended arm during an
# active pick can no longer generate false positives that matter, even
# if it's still visible. If a folded (not extended) arm still intrudes
# during SEARCH itself, that's the next thing to check -- the debug
# window now prints the current state name on every frame specifically
# so this can be diagnosed from a recording without guessing.
CAMERA_MOUNT_TRANSLATION = (0.0, 0.0, 0.85)
CAMERA_MOUNT_ROTATION_AXIS_ANGLE = (0.0, 1.0, 0.0, 1.35)  # ~77 degrees down from horizontal, near-overhead

# Arm base (ARM1 joint origin) offset in the same BODY local frame --
# from the real Youbot.proto: arm Solid at (0.156, 0, 0), ARM1 hinge
# anchor at a further local (0, 0, 0.077) inside that Solid.
ARM_BASE_OFFSET_BODY_FRAME = (0.156, 0.0, 0.077)

APPROACH_STOP_DEPTH = 0.32       # meters -- switch from base-driving to arm-picking once this close
TURN_GAIN = 1.2
FORWARD_SPEED = 0.08             # m/s, deliberately slow -- this base has no obstacle avoidance
SEARCH_OMEGA = 0.6               # rad/s in-place rotation when nothing is detected
LIFT_POSE = (0.05, 0.20, 0.15)   # arm-frame (lateral, forward, height) -- a safe raised pose
DEPOSIT_POSE = (0.20, 0.05, 0.10)  # arm-frame pose over the onboard bin
STEPS_TO_SETTLE = 15             # simulation steps to wait after commanding a joint move, before acting on it


def body_to_arm_frame(x_body, y_body, z_body):
    """Body frame (x=forward, y=left, z=up) -> YoubotArm.move_to's own
    (x=lateral, y=forward, z=height-from-ARM1) convention. See
    youbot_control.py's YoubotArm docstring and the derivation in this
    project's notes: arm_ik's formulas only make sense with y as the
    forward/reference axis (alpha = -asin(x/hypot(x,y)) is undefined
    unless y is what "straight ahead" is measured against), and a
    positive ARM1 angle (positive Z-axis rotation, body x->y i.e.
    forward->left) must correspond to NEGATIVE arm_ik-x given
    alpha=-asin(x/y1) -- so lateral-right maps to positive arm_ik-x.
    This is the single piece of this controller most likely to need a
    sign flip after the first live run; if the arm reaches to the
    mirror-opposite side of the detected leaf, negate the first
    (lateral) term below.
    """
    ox, oy, oz = ARM_BASE_OFFSET_BODY_FRAME
    forward = x_body - ox
    left = y_body - oy
    height = z_body - oz
    lateral_right = -left
    return lateral_right, forward, height


class LeafCollectorController:
    STATE_SEARCH = "SEARCH"
    STATE_APPROACH = "APPROACH"
    STATE_PICK = "PICK"
    STATE_LIFT = "LIFT"
    STATE_DEPOSIT = "DEPOSIT"
    STATE_RESET = "RESET"

    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        self.range_finder = self.robot.getDevice("range-finder")
        self.range_finder.enable(self.timestep)

        self.arm = YoubotArm(self.robot)
        self.gripper = YoubotGripper(self.robot)
        self.base = YoubotBase(self.robot)
        self.connector = self.robot.getDevice("gripper connector")
        self.connector.enablePresence(self.timestep)

        self.vision = LeafVisionBridge(WEIGHTS_PATH, conf=CONF_THRESHOLD)
        print(f"[youbot_leaf_collector] loaded real leaf model from {WEIGHTS_PATH}")

        self.projector = RgbdProjector(
            CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV,
            CAMERA_MOUNT_TRANSLATION, CAMERA_MOUNT_ROTATION_AXIS_ANGLE,
        )

        self.arm.set_joint_reset_pose()
        self.state = self.STATE_SEARCH
        self.state_step_count = 0
        self.leaves_collected = 0

    def _enter_state(self, state):
        self.state = state
        self.state_step_count = 0

    def _annotate_and_show(self, frame, detections, label_suffix=""):
        annotated = self.vision.annotate(frame, detections) if detections is not None else frame
        cv2.putText(annotated, self.state + label_suffix, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("leaf_v3 -- youBot overhead camera", annotated)
        cv2.waitKey(1)

    def step(self) -> bool:
        if self.robot.step(self.timestep) == -1:
            return False

        frame = camera_image_to_bgr(self.camera)

        # Detection only matters for SEARCH/APPROACH (see the state
        # handlers below) -- running it during PICK/LIFT/DEPOSIT/RESET
        # wastes inference time and, worse, shows a confusing wall of
        # false positives on the arm's own close-up mesh in the debug
        # window with zero effect on behavior. The arm has never once
        # appeared in this model's training data (Track B's synthetic
        # scenes are leaves/waste on ground only), so at close range its
        # orange plastic gets misread as a dry leaf -- expected, not a
        # sign best.pt is broken; see this project's simulation/README.md.
        detects_this_state = self.state in (self.STATE_SEARCH, self.STATE_APPROACH)
        detections = self.vision.detect(frame) if detects_this_state else []
        self.state_step_count += 1

        if self.state == self.STATE_SEARCH:
            self._do_search(detections)
        elif self.state == self.STATE_APPROACH:
            self._do_approach(detections)
        elif self.state == self.STATE_PICK:
            self._do_pick(detections)
        elif self.state == self.STATE_LIFT:
            self._do_lift()
        elif self.state == self.STATE_DEPOSIT:
            self._do_deposit()
        elif self.state == self.STATE_RESET:
            self._do_reset()

        # Shown/labeled AFTER the state handlers above run, so the overlay
        # reflects the state as of the END of this step -- e.g. a leaf
        # found while in SEARCH shows the box under the "APPROACH" label
        # it just transitioned into, not the stale "SEARCH" label from
        # before that transition happened.
        self._annotate_and_show(frame, detections if detects_this_state else None,
                                 label_suffix="" if detects_this_state else " (detection paused)")

        return True

    # ---- state handlers --------------------------------------------------
    def _do_search(self, detections):
        if detections:
            self._enter_state(self.STATE_APPROACH)
            return
        self.base.move(0.0, 0.0, SEARCH_OMEGA)

    def _do_approach(self, detections):
        if not detections:
            self._enter_state(self.STATE_SEARCH)
            return

        target = max(detections, key=lambda d: d.conf)
        x1, y1, x2, y2 = target.box
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2
        offset = (box_cx - CAMERA_WIDTH / 2) / (CAMERA_WIDTH / 2)

        depth_map = self.range_finder.getRangeImage()
        px, py = int(box_cx), int(box_cy)
        depth = depth_map[py * CAMERA_WIDTH + px]

        if not math.isfinite(depth) or depth <= 0:
            # depth sensor gave nothing usable at this pixel (e.g. sensor
            # max range exceeded) -- keep approaching cautiously on vision alone
            self.base.move(FORWARD_SPEED * 0.5, 0.0, -TURN_GAIN * offset)
            return

        print(f"[youbot_leaf_collector] tracking leaf, confidence={target.conf:.2f}, depth={depth:.2f}m")

        if depth <= APPROACH_STOP_DEPTH:
            self.base.stop()
            self._pending_pixel = (box_cx, box_cy)
            self._enter_state(self.STATE_PICK)
            return

        self.base.move(FORWARD_SPEED, 0.0, -TURN_GAIN * offset)

    def _do_pick(self, detections):
        if self.state_step_count == 1:
            box_cx, box_cy = self._pending_pixel
            depth_map = self.range_finder.getRangeImage()
            px, py = int(box_cx), int(box_cy)
            depth = depth_map[py * CAMERA_WIDTH + px]

            x_body, y_body, z_body = self.projector.pixel_to_body_frame(box_cx, box_cy, depth)
            arm_x, arm_y, arm_z = body_to_arm_frame(x_body, y_body, z_body)
            print(f"[youbot_leaf_collector] real depth-projected target: "
                  f"body=({x_body:.3f},{y_body:.3f},{z_body:.3f}) "
                  f"arm=({arm_x:.3f},{arm_y:.3f},{arm_z:.3f})")
            try:
                self.arm.move_to(arm_x, arm_y, arm_z)
                self.gripper.release()
            except ValueError as e:
                print(f"[youbot_leaf_collector] target unreachable, aborting pick: {e}")
                self._enter_state(self.STATE_SEARCH)
            return

        if self.state_step_count == STEPS_TO_SETTLE:
            self.gripper.grip()
            return

        if self.state_step_count == STEPS_TO_SETTLE + STEPS_TO_SETTLE:
            if self.connector.getPresence() == 1:
                self.connector.lock()
                self.leaves_collected += 1
                print(f"[youbot_leaf_collector] collected leaf #{self.leaves_collected} "
                      f"(real Connector lock, real physics)")
            else:
                print("[youbot_leaf_collector] gripper closed but no Connector presence -- missed grasp")
            self._enter_state(self.STATE_LIFT)

    def _do_lift(self):
        if self.state_step_count == 1:
            self.arm.move_to(*LIFT_POSE)
        if self.state_step_count >= STEPS_TO_SETTLE:
            self._enter_state(self.STATE_DEPOSIT)

    def _do_deposit(self):
        if self.state_step_count == 1:
            self.arm.move_to(*DEPOSIT_POSE)
        if self.state_step_count == STEPS_TO_SETTLE:
            self.gripper.release()
            self.connector.unlock()
        if self.state_step_count >= STEPS_TO_SETTLE + 10:
            self._enter_state(self.STATE_RESET)

    def _do_reset(self):
        if self.state_step_count == 1:
            self.arm.set_joint_reset_pose()
        if self.state_step_count >= STEPS_TO_SETTLE:
            self._enter_state(self.STATE_SEARCH)


controller = LeafCollectorController()
while controller.step():
    pass
