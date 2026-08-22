"""Python port of Cyberbotics' own youBot control library
(webots/projects/robots/kuka/youbot/libraries/youbot_control, C source),
kept faithful to the original formulas rather than reimplemented from
scratch — this is the same analytical inverse-kinematics solver and
mecanum-wheel mixing Cyberbotics ships and tests for this exact robot,
just called from Python instead of C so it can share a process with
ultralytics/opencv. Link lengths, joint signs, and the IK derivation are
copied as-is from arm.c; only the language changed.

Real, deterministic geometry -- not fitted, not approximate for the sake
of a demo. If this is wrong it's because the port has a bug, not because
the underlying kinematics were invented for this project.
"""

import math

# ---- real youBot arm link lengths, meters (arm.c: arm_get_sub_arm_length) --
ARM_LINK = {
    "ARM1": 0.253,
    "ARM2": 0.155,
    "ARM3": 0.135,
    "ARM4": 0.081,
    "ARM5": 0.105,
}

# ---- real youBot base geometry, meters (base.c) ---------------------------
WHEEL_RADIUS = 0.05
LX = 0.228  # longitudinal distance from base center to a wheel
LY = 0.158  # lateral distance from base center to a wheel


class YoubotArm:
    """Real analytical IK for the 5-DOF youBot arm, ported from arm_ik()
    in arm.c. Coordinates (x, y, z) are in the arm's own base frame (the
    ARM1 hinge joint's origin) -- see the mounting-offset constants in
    youbot_leaf_collector.py for how a world/body-frame target gets
    converted into this frame before being passed here."""

    def __init__(self, robot):
        self.arm1 = robot.getDevice("arm1")
        self.arm2 = robot.getDevice("arm2")
        self.arm3 = robot.getDevice("arm3")
        self.arm4 = robot.getDevice("arm4")
        self.arm5 = robot.getDevice("arm5")
        self.arm2.setVelocity(0.5)  # matches arm_init()'s deliberate slow-down on ARM2

    def set_joint_reset_pose(self):
        """Matches arm.c's ARM_RESET preset -- a safe folded pose to start from."""
        self.arm1.setPosition(0.0)
        self.arm2.setPosition(1.57)
        self.arm3.setPosition(-2.635)
        self.arm4.setPosition(1.78)
        self.arm5.setPosition(0.0)

    def move_to(self, x: float, y: float, z: float):
        """Direct port of arm_ik(x, y, z) from arm.c."""
        a = ARM_LINK["ARM2"]
        b = ARM_LINK["ARM3"]

        y1 = math.sqrt(x * x + y * y)
        z1 = z + ARM_LINK["ARM4"] + ARM_LINK["ARM5"] - ARM_LINK["ARM1"]
        c = math.sqrt(y1 * y1 + z1 * z1)

        # Same reachability guard the C version doesn't have but needs --
        # asin/acos below are undefined outside [-1, 1], which a
        # depth-sensor-derived target can absolutely produce if the
        # detection or depth reading is off. Fail loudly rather than
        # silently commanding NaN joint positions.
        if y1 == 0 or not (-1.0 <= x / y1 <= 1.0):
            raise ValueError(f"target ({x:.3f},{y:.3f},{z:.3f}) has no valid azimuth solution")
        cos_beta_term = (a * a + c * c - b * b) / (2.0 * a * c)
        cos_gamma_term = (a * a + b * b - c * c) / (2.0 * a * b)
        if not (-1.0 <= cos_beta_term <= 1.0) or not (-1.0 <= cos_gamma_term <= 1.0):
            raise ValueError(f"target ({x:.3f},{y:.3f},{z:.3f}) is outside the arm's reach")

        alpha = -math.asin(x / y1)
        beta = -(math.pi / 2 - math.acos(cos_beta_term) - math.atan(z1 / y1))
        gamma = -(math.pi - math.acos(cos_gamma_term))
        delta = -(math.pi + (beta + gamma))
        epsilon = math.pi / 2 + alpha

        self.arm1.setPosition(alpha)
        self.arm2.setPosition(beta)
        self.arm3.setPosition(gamma)
        self.arm4.setPosition(delta)
        self.arm5.setPosition(epsilon)


class YoubotGripper:
    """Real two-finger gripper control, ported from gripper.c. Drives
    both fingers explicitly (the original only drives finger::left and
    relies on a mechanical coupling in some youBot variants; driving both
    independently is correct regardless and doesn't depend on that)."""

    MIN_POS = 0.0
    MAX_POS = 0.025

    def __init__(self, robot):
        self.left = robot.getDevice("finger::left")
        self.right = robot.getDevice("finger::right")
        self.left.setVelocity(0.03)
        self.right.setVelocity(0.03)

    def grip(self):
        self.left.setPosition(self.MIN_POS)
        self.right.setPosition(self.MIN_POS)

    def release(self):
        self.left.setPosition(self.MAX_POS)
        self.right.setPosition(self.MAX_POS)


class YoubotBase:
    """Real omnidirectional mecanum-wheel base control, ported from
    base_move() in base.c."""

    def __init__(self, robot):
        self.wheels = [robot.getDevice(f"wheel{i + 1}") for i in range(4)]
        for wheel in self.wheels:
            wheel.setPosition(float("inf"))
            wheel.setVelocity(0.0)

    def move(self, vx: float, vy: float, omega: float):
        k = LX + LY
        speeds = [
            (vx + vy + k * omega) / WHEEL_RADIUS,
            (vx - vy - k * omega) / WHEEL_RADIUS,
            (vx - vy + k * omega) / WHEEL_RADIUS,
            (vx + vy - k * omega) / WHEEL_RADIUS,
        ]
        for wheel, speed in zip(self.wheels, speeds):
            wheel.setVelocity(speed)

    def stop(self):
        self.move(0.0, 0.0, 0.0)
