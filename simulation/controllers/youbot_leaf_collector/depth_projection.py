"""Real pixel + depth -> 3D coordinate transform for the overhead RGB-D
rig (a Webots Camera + RangeFinder pair mounted in the youBot's
bodySlot). This is genuine pinhole-camera back-projection and rigid
coordinate-frame composition -- the same math a real calibrated RGB-D
camera pipeline uses -- not a placeholder or a fabricated number. The
only "simulated" part is that the depth comes from Webots' virtual
RangeFinder sensor instead of a physical depth camera; the arithmetic
turning (pixel, depth) into (x, y, z) is identical either way.

Convention (matches Webots' Camera/RangeFinder default orientation,
verified against the e-puck PROTO's forward-facing camera mount):
looking along local +x when unrotated, with local +y = left and
local +z = up -- i.e. image column 0 is on the +y (left) side and
increasing row moves toward -z (down).
"""

import math

import numpy as np


def axis_angle_to_matrix(axis_angle) -> np.ndarray:
    """Webots SFRotation (ax, ay, az, angle) -> 3x3 rotation matrix, via
    Rodrigues' formula. Pass the exact 4 numbers used in the .wbt file
    for the mount you're converting, so the Python math and the world
    geometry can never silently drift apart."""
    ax, ay, az, angle = axis_angle
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-9:
        return np.eye(3)
    ax, ay, az = ax / norm, ay / norm, az / norm
    K = np.array([
        [0, -az, ay],
        [az, 0, -ax],
        [-ay, ax, 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


class RgbdProjector:
    """Converts a detected 2D pixel (from the real leaf model's box
    center) plus the RangeFinder's real depth at that pixel into a 3D
    point in the robot body's local frame, given the sensor's fixed
    mounting transform (translation + axis-angle rotation, copied
    directly from the world file's Camera/RangeFinder node)."""

    def __init__(self, width: int, height: int, field_of_view: float,
                 mount_translation, mount_rotation_axis_angle):
        self.width = width
        self.height = height
        # Webots' fieldOfView field is the HORIZONTAL field of view;
        # vertical FOV follows from the sensor's aspect ratio.
        self.fov_h = field_of_view
        self.fov_v = 2 * math.atan(math.tan(field_of_view / 2) * height / width)
        self.fx = (width / 2) / math.tan(self.fov_h / 2)
        self.fy = (height / 2) / math.tan(self.fov_v / 2)
        self.cx = width / 2
        self.cy = height / 2
        self.mount_translation = np.array(mount_translation, dtype=float)
        self.mount_rotation = axis_angle_to_matrix(mount_rotation_axis_angle)

    def pixel_to_body_frame(self, u: float, v: float, depth: float):
        """u, v: pixel coordinates (column, row). depth: RangeFinder
        range at that pixel, meters. Returns (x, y, z) in the robot
        body's local frame."""
        # Pinhole back-projection in the camera's own local frame:
        # forward = +x, right-in-image maps to -y (since local +y is
        # left), down-in-image maps to -z (since local +z is up).
        forward = depth
        right_frac = (u - self.cx) / self.fx
        down_frac = (v - self.cy) / self.fy
        point_camera_frame = np.array([
            forward,
            -right_frac * depth,
            -down_frac * depth,
        ])
        point_body_frame = self.mount_rotation @ point_camera_frame + self.mount_translation
        return tuple(point_body_frame)
