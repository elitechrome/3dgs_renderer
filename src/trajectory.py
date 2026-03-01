"""Camera trajectory with keyframe management and spline interpolation.

Stores camera keyframes in Viser convention (position, wxyz quaternion, vertical FOV)
and exports to the INRIA JSON format consumed by the Vulkan renderer.
"""

import json
import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicSpline


@dataclass
class CameraKeyframe:
    """A single camera keyframe in Viser convention.

    Attributes:
        position: (3,) camera world position.
        wxyz: (4,) orientation quaternion (w, x, y, z) — C2W rotation.
              The camera looks along its local -Z axis (OpenGL convention).
        fov_y: Vertical field-of-view in radians.
    """
    position: np.ndarray
    wxyz: np.ndarray
    fov_y: float


class CameraTrajectory:
    """Ordered list of camera keyframes with spline interpolation."""

    def __init__(self):
        self.keyframes: list[CameraKeyframe] = []

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_keyframe(self, position, wxyz, fov_y):
        self.keyframes.append(CameraKeyframe(
            position=np.asarray(position, dtype=np.float64),
            wxyz=np.asarray(wxyz, dtype=np.float64),
            fov_y=float(fov_y),
        ))

    def remove_keyframe(self, index: int):
        if -len(self.keyframes) <= index < len(self.keyframes):
            self.keyframes.pop(index)

    def clear(self):
        self.keyframes.clear()

    def move_keyframe(self, src: int, dst: int):
        """Move keyframe at *src* to position *dst*."""
        kf = self.keyframes.pop(src)
        self.keyframes.insert(dst, kf)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate(self, t: float) -> CameraKeyframe:
        """Return an interpolated keyframe at normalised time *t* ∈ [0, 1].

        Position is cubic-spline interpolated; orientation is Slerp-ed;
        FOV is linearly interpolated.
        """
        n = len(self.keyframes)
        if n == 0:
            raise ValueError("No keyframes in trajectory")
        if n == 1:
            return self.keyframes[0]

        t = float(np.clip(t, 0.0, 1.0))
        times = np.linspace(0.0, 1.0, n)

        # --- Position: cubic spline ---
        positions = np.array([kf.position for kf in self.keyframes])
        if n >= 3:
            cs = CubicSpline(times, positions, bc_type="clamped")
        else:
            cs = CubicSpline(times, positions)
        pos = cs(t)

        # --- Orientation: Slerp ---
        quats_scipy = [
            Rotation.from_quat([kf.wxyz[1], kf.wxyz[2], kf.wxyz[3], kf.wxyz[0]])
            for kf in self.keyframes
        ]
        slerp = Slerp(times, Rotation.concatenate(quats_scipy))
        rot = slerp(t)
        xyzw = rot.as_quat()  # scipy convention: (x, y, z, w)
        wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

        # --- FOV: linear ---
        fovs = np.array([kf.fov_y for kf in self.keyframes])
        fov = float(np.interp(t, times, fovs))

        return CameraKeyframe(position=pos, wxyz=wxyz, fov_y=fov)

    # ------------------------------------------------------------------
    # Bulk generation
    # ------------------------------------------------------------------

    def to_inria_cameras(self, num_frames: int, width: int, height: int) -> list[dict]:
        """Sample *num_frames* cameras along the trajectory in INRIA format."""
        cameras = []
        for i in range(num_frames):
            t = 0.0 if num_frames == 1 else i / (num_frames - 1)
            kf = self.interpolate(t)
            cam = keyframe_to_inria(kf, width, height)
            cam["id"] = i
            cam["img_name"] = f"frame_{i:05d}"
            cameras.append(cam)
        return cameras

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        data = {
            "keyframes": [
                {
                    "position": kf.position.tolist(),
                    "wxyz": kf.wxyz.tolist(),
                    "fov_y": kf.fov_y,
                }
                for kf in self.keyframes
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.keyframes = [
            CameraKeyframe(
                position=np.asarray(kf["position"], dtype=np.float64),
                wxyz=np.asarray(kf["wxyz"], dtype=np.float64),
                fov_y=float(kf["fov_y"]),
            )
            for kf in data["keyframes"]
        ]

    @property
    def num_keyframes(self) -> int:
        return len(self.keyframes)


# ------------------------------------------------------------------
# Conversion helpers
# ------------------------------------------------------------------

def keyframe_to_inria(kf: CameraKeyframe, width: int, height: int) -> dict:
    """Convert a CameraKeyframe (Viser convention) to the INRIA camera dict
    expected by the Vulkan renderer's ``importCamerasINRIA``.

    The existing pipeline stores a *mixed* view-matrix
    (``R_c2w | T_w2c``) and then inverts it to produce the JSON fields.
    We replicate exactly that convention here so the C++ side interprets
    the values correctly.
    """
    R_c2w = Rotation.from_quat(
        [kf.wxyz[1], kf.wxyz[2], kf.wxyz[3], kf.wxyz[0]]
    ).as_matrix()
    R_w2c = R_c2w.T
    T_w2c = -R_w2c @ kf.position

    # Build the same mixed view-matrix that main.py constructs
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R_c2w
    view_matrix[:3, 3] = T_w2c

    c2w = np.linalg.inv(view_matrix)

    # Focal lengths from vertical FOV
    fy = height / (2.0 * np.tan(kf.fov_y / 2.0))
    fx = fy  # square pixels

    return {
        "position": c2w[:3, 3].tolist(),
        "rotation": c2w[:3, :3].tolist(),
        "fx": float(fx),
        "fy": float(fy),
        "width": int(width),
        "height": int(height),
    }


def inria_to_keyframe(cam: dict) -> CameraKeyframe:
    """Recover a CameraKeyframe from an INRIA camera dict.

    This is the inverse of :func:`keyframe_to_inria` — it reconstructs
    the Viser-convention camera from the JSON representation.
    """
    inria_pos = np.asarray(cam["position"], dtype=np.float64)
    inria_rot = np.asarray(cam["rotation"], dtype=np.float64)

    # The INRIA "rotation" is the R_w2c stored after inversion of the
    # mixed view-matrix.  The C2W rotation is its inverse (= transpose).
    R_c2w = inria_rot.T  # == R_w2c.T
    # The INRIA "position" equals -R_w2c @ T_w2c.  Recover T_w2c first,
    # then world position.
    T_w2c = -inria_rot.T @ inria_pos  # -R_w2c^-1 @ inria_pos = -R_c2w @ inria_pos
    # Actually: T_w2c = -R_w2c @ cam_world_pos  =>  cam_world_pos = -R_w2c^T @ T_w2c
    # But we need to recover cam_world_pos from inria_pos = -R_w2c @ T_w2c.
    # A simpler path: rebuild the mixed view-matrix and read out cam_pos.
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R_c2w
    # T_w2c from: c2w = inv(view_matrix), c2w[:3,3] = inria_pos
    # => -R_c2w.T @ T_w2c = inria_pos  =>  T_w2c = -R_c2w @ inria_pos
    T_w2c_recovered = -R_c2w @ inria_pos
    view_matrix[:3, 3] = T_w2c_recovered

    # Now recover world position: cam_pos = -R_w2c^T @ T_w2c = -R_c2w @ T_w2c
    cam_pos = -R_c2w @ T_w2c_recovered

    # Quaternion (wxyz)
    xyzw = Rotation.from_matrix(R_c2w).as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    # FOV from fy
    fy = cam["fy"]
    height = cam["height"]
    fov_y = 2.0 * np.arctan(height / (2.0 * fy))

    return CameraKeyframe(position=cam_pos, wxyz=wxyz, fov_y=fov_y)
