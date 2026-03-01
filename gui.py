#!/usr/bin/env python3
"""Interactive camera trajectory editor with RGB + depth preview.

Launches a Viser web GUI (opens in browser) where you can:
  1. Navigate a 3-D viewport showing the point cloud / Gaussian splats.
  2. Add camera keyframes from the current view.
  3. Scrub a timeline slider and play back the trajectory.
  4. Render RGB + depth previews at any frame on the fly.
  5. Batch-render all frames and export cameras.json.

Usage:
    python gui.py --ply path/to/scene.ply [--config config.yaml] [--port 8080]
"""

from __future__ import annotations

import os
import sys
import time
import threading
import argparse
from pathlib import Path

import numpy as np
import yaml
import viser

# Ensure our ``src/`` package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from trajectory import (
    CameraTrajectory,
    CameraKeyframe,
    keyframe_to_inria,
    inria_to_keyframe,
)
from preview_renderer import PreviewRenderer
from inria_io import save_inria_cameras
from depth_utils import depth_to_colormap


# ======================================================================
# Helpers
# ======================================================================

def _find_binary(override: str | None = None) -> str:
    """Locate the Vulkan renderer binary."""
    if override and os.path.isfile(override):
        return override
    candidates = [
        "vk_gaussian_splatting/_bin/Release/vk_gaussian_splatting",
        "vk_gaussian_splatting/build/vk_gaussian_splatting",
        "vk_gaussian_splatting/_bin/vk_gaussian_splatting",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Cannot find vk_gaussian_splatting binary. "
        "Build it first or pass --binary <path>."
    )


def _load_ply_gaussians(ply_path: str, max_splats: int = 500_000):
    """Load full Gaussian splat data from a 3DGS PLY file.

    Returns a dict with keys ``centers``, ``covariances``, ``rgbs``,
    ``opacities`` ready for ``viser.scene.add_gaussian_splats()``, or
    *None* on failure.

    The PLY stores log-space scales, unit quaternions, logit opacities,
    and SH DC coefficients — all converted here.
    """
    try:
        from plyfile import PlyData
    except ImportError:
        print("[gui] plyfile not installed — cannot load Gaussians.")
        return None

    try:
        ply = PlyData.read(ply_path)
        v = ply["vertex"]
        n = len(v["x"])

        # Subsample if the scene is very large
        idx = np.arange(n)
        if n > max_splats:
            idx = np.random.default_rng(42).choice(n, max_splats, replace=False)
            idx.sort()
        m = len(idx)

        # ---- Centres (N, 3) ----
        centers = np.column_stack(
            [v["x"][idx], v["y"][idx], v["z"][idx]]
        ).astype(np.float32)

        # ---- RGB from DC SH (N, 3) in [0, 1] ----
        C0 = 0.28209479177387814
        try:
            r = np.clip(v["f_dc_0"][idx] * C0 + 0.5, 0, 1)
            g = np.clip(v["f_dc_1"][idx] * C0 + 0.5, 0, 1)
            b = np.clip(v["f_dc_2"][idx] * C0 + 0.5, 0, 1)
            rgbs = np.column_stack([r, g, b]).astype(np.float32)
        except (ValueError, KeyError):
            rgbs = np.full((m, 3), 0.7, dtype=np.float32)

        # ---- Opacity: sigmoid(logit) → (N, 1) ----
        logit_opacity = np.array(v["opacity"][idx], dtype=np.float32)
        opacities = (1.0 / (1.0 + np.exp(-logit_opacity))).reshape(-1, 1)

        # ---- Covariance matrices (N, 3, 3) from scale + quaternion ----
        #  scale stored as log(scale), quaternion as (w, x, y, z)
        scales = np.column_stack([
            np.array(v["scale_0"][idx], dtype=np.float32),
            np.array(v["scale_1"][idx], dtype=np.float32),
            np.array(v["scale_2"][idx], dtype=np.float32),
        ])
        scales = np.exp(scales)  # log-space → linear

        quats = np.column_stack([
            np.array(v["rot_0"][idx], dtype=np.float32),
            np.array(v["rot_1"][idx], dtype=np.float32),
            np.array(v["rot_2"][idx], dtype=np.float32),
            np.array(v["rot_3"][idx], dtype=np.float32),
        ])
        # Normalise quaternions
        quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12)

        covariances = _build_covariances(quats, scales)

        return {
            "centers": centers,
            "covariances": covariances,
            "rgbs": rgbs,
            "opacities": opacities,
        }
    except Exception as exc:
        print(f"[gui] Warning: could not read PLY for Gaussian splats: {exc}")
        import traceback; traceback.print_exc()
        return None


def _build_covariances(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Build (N, 3, 3) covariance matrices from quaternions and scales.

    Cov = R @ diag(s^2) @ R^T  where R is the rotation from the quaternion
    and s is the per-axis scale.

    Args:
        quats: (N, 4)  quaternions  (w, x, y, z).
        scales: (N, 3) axis scales (linear, not log).
    """
    n = len(quats)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    # Rotation matrix columns from quaternion (row-major R)
    R = np.zeros((n, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    # S = diag(scale^2)
    s2 = scales ** 2  # (N, 3)

    # Cov = R @ diag(s^2) @ R^T  — expand manually for speed
    # Cov_ij = sum_k R_ik * s2_k * R_jk
    cov = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            cov[:, i, j] = np.sum(R[:, i, :] * s2 * R[:, j, :], axis=1)

    return cov


# ======================================================================
# Main GUI class
# ======================================================================

class TrajectoryGUI:
    """Viser-based interactive camera trajectory editor."""

    def __init__(
        self,
        ply_path: str,
        config_path: str = "config.yaml",
        binary_path: str | None = None,
        port: int = 8080,
        output_dir: str = "output",
    ):
        # --- Config ---
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        cam_cfg = self.config.get("camera", {})
        self.width = cam_cfg.get("width", 1024)
        self.height = cam_cfg.get("height", 768)

        fy = cam_cfg.get("fy", cam_cfg.get("fx", self.width / 2))
        self.default_fov_y = float(2.0 * np.arctan(self.height / (2.0 * fy)))

        self.ply_path = os.path.abspath(ply_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Trajectory ---
        self.trajectory = CameraTrajectory()

        # --- Renderer ---
        binary = _find_binary(binary_path)
        self.renderer = PreviewRenderer(binary, self.ply_path, self.width, self.height)

        # --- Playback state ---
        self._playing = False
        self._play_thread: threading.Thread | None = None

        # --- Image caches (for GUI display) ---
        self._last_rgb: np.ndarray | None = None
        self._last_depth: np.ndarray | None = None
        self._last_depth_cm: np.ndarray | None = None

        # --- Rendering lock (one render at a time) ---
        self._render_lock = threading.Lock()

        # --- Viser server ---
        self.server = viser.ViserServer(port=port)
        self.server.gui.configure_theme(
            control_layout="fixed",
            control_width="medium",
            dark_mode=True,
            show_logo=False,
        )
        self.server.scene.set_up_direction("-y")

        self._build_gui()
        self._load_point_cloud()
        self._update_scene()

    # ==================================================================
    # GUI construction
    # ==================================================================

    def _build_gui(self):
        s = self.server

        # --- Keyframe controls ---
        with s.gui.add_folder("Keyframes", order=0):
            self._btn_add_kf = s.gui.add_button(
                "Add Keyframe", color="green", icon="plus", order=0
            )
            self._btn_remove_last = s.gui.add_button(
                "Remove Last", color="red", icon="minus", order=1
            )
            self._btn_clear = s.gui.add_button(
                "Clear All", icon="trash", order=2
            )
            self._txt_kf_count = s.gui.add_text(
                "Count", initial_value="0", disabled=True, order=3,
            )

        # --- Timeline / playback ---
        with s.gui.add_folder("Timeline", order=1):
            self._slider_frame = s.gui.add_slider(
                "Frame", min=0, max=99, step=1, initial_value=0, order=0,
            )
            self._num_total = s.gui.add_number(
                "Total Frames", initial_value=100, min=2, max=2000, step=1, order=1,
            )
            self._btn_play = s.gui.add_button("Play", icon="player-play", order=2)
            self._btn_pause = s.gui.add_button(
                "Pause", icon="player-pause", order=3, visible=False,
            )
            self._slider_rate = s.gui.add_slider(
                "Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0, order=4,
            )
            self._num_fps = s.gui.add_number(
                "FPS", initial_value=24, min=1, max=60, step=1, order=5,
            )

        # --- Rendering ---
        with s.gui.add_folder("Render", order=2):
            self._btn_render = s.gui.add_button(
                "Render Frame", color="blue", icon="camera", order=0,
            )
            self._dropdown_display = s.gui.add_dropdown(
                "Display",
                options=("RGB", "Depth", "Side-by-Side"),
                initial_value="Side-by-Side",
                order=1,
            )
            self._btn_render_all = s.gui.add_button(
                "Render All Frames", icon="movie", order=2,
            )
            self._progress = s.gui.add_progress_bar(0.0, visible=False, order=3)
            self._img_preview = s.gui.add_image(
                np.zeros((64, 128, 3), dtype=np.uint8),
                label="Preview",
                order=4,
                visible=False,
            )

        # --- Export / import ---
        with s.gui.add_folder("Export / Import", order=3):
            self._btn_save_traj = s.gui.add_button(
                "Save Trajectory", icon="device-floppy", order=0,
            )
            self._btn_load_traj = s.gui.add_button(
                "Load Trajectory", icon="folder", order=1,
            )
            self._btn_export = s.gui.add_button(
                "Export cameras.json", icon="file-export", order=2,
            )
            self._btn_batch_export = s.gui.add_button(
                "Batch Render + Export", color="blue", icon="rocket", order=3,
            )

        # --- Wire callbacks ---
        self._wire_callbacks()

    def _wire_callbacks(self):
        # Keyframes
        @self._btn_add_kf.on_click
        def _(_):
            self._add_keyframe_from_view()

        @self._btn_remove_last.on_click
        def _(_):
            self.trajectory.remove_keyframe(-1)
            self._update_scene()

        @self._btn_clear.on_click
        def _(_):
            self.trajectory.clear()
            self._update_scene()

        # Timeline
        @self._slider_frame.on_update
        def _(_):
            if self.trajectory.num_keyframes >= 2:
                self._move_to_frame(int(self._slider_frame.value))

        @self._num_total.on_update
        def _(_):
            self._slider_frame.max = int(self._num_total.value) - 1

        # Play / pause
        @self._btn_play.on_click
        def _(_):
            self._start_playback()

        @self._btn_pause.on_click
        def _(_):
            self._stop_playback()

        # Render
        @self._btn_render.on_click
        def _(_):
            threading.Thread(target=self._render_current_frame, daemon=True).start()

        @self._btn_render_all.on_click
        def _(_):
            threading.Thread(target=self._render_all_frames, daemon=True).start()

        # Export / import
        @self._btn_save_traj.on_click
        def _(_):
            path = self.output_dir / "trajectory.json"
            self.trajectory.save(str(path))
            print(f"[gui] Trajectory saved to {path}")

        @self._btn_load_traj.on_click
        def _(_):
            path = self.output_dir / "trajectory.json"
            if path.exists():
                self.trajectory.load(str(path))
                self._update_scene()
                print(f"[gui] Trajectory loaded from {path}")
            else:
                print(f"[gui] No trajectory file at {path}")

        @self._btn_export.on_click
        def _(_):
            self._export_cameras_json()

        @self._btn_batch_export.on_click
        def _(_):
            threading.Thread(target=self._batch_render_and_export, daemon=True).start()

    # ==================================================================
    # Keyframes
    # ==================================================================

    def _add_keyframe_from_view(self):
        """Capture the current viewer camera as a keyframe."""
        clients = self.server.get_clients()
        if not clients:
            print("[gui] No connected clients — open the browser first.")
            return
        client = next(iter(clients.values()))
        cam = client.camera
        self.trajectory.add_keyframe(
            position=np.array(cam.position, dtype=np.float64),
            wxyz=np.array(cam.wxyz, dtype=np.float64),
            fov_y=float(cam.fov),
        )
        self._update_scene()

    # ==================================================================
    # Scene visualisation
    # ==================================================================

    def _update_scene(self):
        """Redraw camera frustums and trajectory spline."""
        n = self.trajectory.num_keyframes
        self._txt_kf_count.value = str(n)

        s = self.server.scene

        # Remove old frustums and trajectory
        for name in list(getattr(self, '_scene_handles', [])):
            try:
                s.remove_by_name(name)
            except Exception:
                pass
        self._scene_handles: list[str] = []

        if n == 0:
            return

        aspect = self.width / self.height

        # Draw frustums at keyframes
        for i, kf in enumerate(self.trajectory.keyframes):
            name = f"/keyframes/{i}"
            s.add_camera_frustum(
                name=name,
                fov=kf.fov_y,
                aspect=aspect,
                scale=0.15,
                wxyz=kf.wxyz,
                position=kf.position,
                color=(30, 180, 30) if i == 0 else (180, 80, 30),
                line_width=2.0,
            )
            self._scene_handles.append(name)

        # Draw spline through keyframe positions
        if n >= 2:
            # Sample interpolated positions for a smooth path
            num_samples = max(n * 20, 100)
            path_points = np.zeros((num_samples, 3))
            for j in range(num_samples):
                t = j / (num_samples - 1)
                kf = self.trajectory.interpolate(t)
                path_points[j] = kf.position
            name = "/trajectory_spline"
            s.add_spline_catmull_rom(
                name=name,
                points=path_points,
                color=(60, 160, 255),
                line_width=2.0,
            )
            self._scene_handles.append(name)

    def _load_point_cloud(self):
        """Load the PLY and display Gaussian splats (or fallback to points)."""
        gs = _load_ply_gaussians(self.ply_path)
        if gs is not None:
            centers = gs["centers"]

            # Centre the splats at world origin so the grid aligns
            centroid = centers.mean(axis=0)
            centers -= centroid
            self._world_offset = centroid.astype(np.float64)
            extent = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
            self._scene_extent = extent
            self._scene_center = np.zeros(3, dtype=np.float64)

            print(f"[gui] Loading {len(centers)} Gaussian splats (shifted by {centroid}) …")
            self.server.scene.add_gaussian_splats(
                name="/splats",
                centers=centers,
                covariances=gs["covariances"],
                rgbs=gs["rgbs"],
                opacities=gs["opacities"],
            )
            print(f"[gui] Gaussian splats loaded.  extent={extent:.2f}")
        else:
            self._scene_center = np.zeros(3, dtype=np.float64)
            self._scene_extent = 5.0
            self._world_offset = np.zeros(3, dtype=np.float64)
            print("[gui] Falling back to point cloud preview.")

        # Grid for spatial reference
        self.server.scene.add_grid(
            "/grid",
            width=10.0,
            height=10.0,
            plane="xz",
        )

        # Set initial camera for new clients
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            cam = client.camera
            cam.look_at = (0.0, 0.0, 0.0)
            cam.up_direction = (0.0, -1.0, 0.0)
            # Place camera at a distance looking at the centred object
            d = self._scene_extent * 0.8
            cam.position = (0.0, -d * 0.35, -d)
            cam.fov = self.default_fov_y

    # ==================================================================
    # Timeline / navigation
    # ==================================================================

    def _move_to_frame(self, frame_idx: int):
        """Move the viewer camera to the interpolated pose at *frame_idx*."""
        total = int(self._num_total.value)
        t = frame_idx / max(total - 1, 1)
        kf = self.trajectory.interpolate(t)
        for client in self.server.get_clients().values():
            client.camera.position = kf.position
            client.camera.wxyz = kf.wxyz
            client.camera.fov = kf.fov_y

    # ==================================================================
    # Playback
    # ==================================================================

    def _start_playback(self):
        if self.trajectory.num_keyframes < 2:
            print("[gui] Need at least 2 keyframes for playback.")
            return
        self._playing = True
        self._btn_play.visible = False
        self._btn_pause.visible = True
        self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._play_thread.start()

    def _stop_playback(self):
        self._playing = False
        self._btn_play.visible = True
        self._btn_pause.visible = False

    def _playback_loop(self):
        total = int(self._num_total.value)
        frame = int(self._slider_frame.value)
        while self._playing:
            fps = max(int(self._num_fps.value), 1)
            rate = float(self._slider_rate.value)
            dt = 1.0 / fps / rate

            frame = (frame + 1) % total
            self._slider_frame.value = frame
            self._move_to_frame(frame)
            time.sleep(dt)

    # ==================================================================
    # Coordinate helpers
    # ==================================================================

    def _kf_to_renderer_cam(self, kf: CameraKeyframe) -> dict:
        """Convert a GUI keyframe to an INRIA camera dict for the Vulkan
        renderer, undoing the centring shift applied to the splats."""
        shifted_kf = CameraKeyframe(
            position=kf.position + self._world_offset,
            wxyz=kf.wxyz,
            fov_y=kf.fov_y,
        )
        return keyframe_to_inria(shifted_kf, self.width, self.height)

    def _trajectory_to_renderer_cams(self, num_frames: int) -> list[dict]:
        """Sample the trajectory and return INRIA cameras with the offset applied."""
        cameras = []
        for i in range(num_frames):
            t = 0.0 if num_frames == 1 else i / (num_frames - 1)
            kf = self.trajectory.interpolate(t)
            cam = self._kf_to_renderer_cam(kf)
            cam["id"] = i
            cam["img_name"] = f"frame_{i:05d}"
            cameras.append(cam)
        return cameras

    # ==================================================================
    # Rendering
    # ==================================================================

    def _render_current_frame(self):
        """Render the current timeline frame (blocking — run in a thread)."""
        if self.trajectory.num_keyframes < 1:
            print("[gui] No keyframes — add at least one.")
            return
        if not self._render_lock.acquire(blocking=False):
            return  # already rendering

        try:
            total = int(self._num_total.value)
            frame = int(self._slider_frame.value)
            t = frame / max(total - 1, 1)
            kf = self.trajectory.interpolate(t)
            cam = self._kf_to_renderer_cam(kf)

            self._progress.value = 0.0
            self._progress.visible = True

            # RGB
            rgb = self.renderer.render_frame(cam, visualize=0)
            self._progress.value = 0.5
            self._last_rgb = rgb

            # Depth
            depth = self.renderer.render_frame(cam, visualize=2)
            self._progress.value = 1.0
            self._last_depth = depth

            self._update_preview_image()
            self._progress.visible = False
        finally:
            self._render_lock.release()

    def _render_all_frames(self):
        """Batch-render every frame along the trajectory."""
        if self.trajectory.num_keyframes < 2:
            print("[gui] Need >=2 keyframes for batch render.")
            return
        if not self._render_lock.acquire(blocking=False):
            return

        try:
            total = int(self._num_total.value)
            cameras = self.trajectory.to_inria_cameras(total, self.width, self.height)

            self._progress.value = 0.0
            self._progress.visible = True

            def on_progress(cur, tot):
                self._progress.value = cur / tot

            # RGB pass
            print(f"[gui] Rendering {total} RGB frames …")
            self.renderer.render_trajectory(cameras, visualize=0, progress_callback=on_progress)

            # Depth pass
            print(f"[gui] Rendering {total} depth frames …")
            self.renderer.render_trajectory(cameras, visualize=2, progress_callback=on_progress)

            self._progress.value = 1.0
            print("[gui] Batch render complete.")
            time.sleep(0.5)
            self._progress.visible = False
        finally:
            self._render_lock.release()

    def _update_preview_image(self):
        """Update the sidebar preview image based on selected display mode."""
        mode = self._dropdown_display.value
        rgb = self._last_rgb
        depth = self._last_depth

        if rgb is None and depth is None:
            return

        if mode == "RGB" and rgb is not None:
            img = rgb
        elif mode == "Depth" and depth is not None:
            img = depth_to_colormap(depth)
        elif mode == "Side-by-Side":
            parts = []
            if rgb is not None:
                parts.append(rgb)
            if depth is not None:
                parts.append(depth_to_colormap(depth))
            if not parts:
                return
            # Resize to same height and concatenate horizontally
            h = min(p.shape[0] for p in parts)
            resized = []
            for p in parts:
                if p.shape[0] != h:
                    scale = h / p.shape[0]
                    new_w = int(p.shape[1] * scale)
                    try:
                        import cv2
                        p = cv2.resize(p, (new_w, h))
                    except ImportError:
                        pass
                resized.append(p)
            img = np.concatenate(resized, axis=1)
        else:
            return

        # Ensure uint8 HWC
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        self._img_preview.image = img
        self._img_preview.visible = True

        # Also set as viewport background for immersive viewing
        self.server.scene.set_background_image(img, format="jpeg", jpeg_quality=85)

    # ==================================================================
    # Export
    # ==================================================================

    def _export_cameras_json(self):
        total = int(self._num_total.value)
        cameras = self._trajectory_to_renderer_cams(total)
        path = self.output_dir / "cameras.json"
        save_inria_cameras(cameras, str(path))
        print(f"[gui] Exported {total} cameras to {path}")

    def _batch_render_and_export(self):
        """Render all frames via main.py's pipeline and export."""
        if self.trajectory.num_keyframes < 2:
            print("[gui] Need >=2 keyframes.")
            return
        if not self._render_lock.acquire(blocking=False):
            return

        try:
            total = int(self._num_total.value)
            cameras = self._trajectory_to_renderer_cams(total)

            # Save cameras.json
            cam_path = self.output_dir / "cameras.json"
            save_inria_cameras(cameras, str(cam_path))

            self._progress.value = 0.0
            self._progress.visible = True

            rgb_dir = self.output_dir / "rgb"
            depth_dir = self.output_dir / "depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)

            from main import generate_cfg, run_renderer

            # RGB pass
            print("[gui] Batch rendering RGB …")
            self._progress.value = 0.1
            rgb_cfg, rgb_seq = generate_cfg(cam_path, rgb_dir, total, visualize=0, ext=".png")
            run_renderer(
                self.renderer.binary_path, self.ply_path, rgb_cfg,
                self.width, self.height, visualize=0, num_sequences=rgb_seq + 1,
            )

            self._progress.value = 0.5

            # Depth pass
            print("[gui] Batch rendering depth …")
            depth_cfg, depth_seq = generate_cfg(cam_path, depth_dir, total, visualize=2, ext=".hdr")
            run_renderer(
                self.renderer.binary_path, self.ply_path, depth_cfg,
                self.width, self.height, visualize=2, num_sequences=depth_seq + 1,
            )
            self._progress.value = 0.9

            # Post-processing
            try:
                from depth_utils import process_depth_directory
                paired_dir = self.output_dir / "paired"
                process_depth_directory(
                    hdr_dir=str(depth_dir),
                    rgb_dir=str(rgb_dir),
                    output_dir=str(paired_dir),
                    depth_format="npy",
                )
            except Exception as exc:
                print(f"[gui] Post-processing warning: {exc}")

            self._progress.value = 1.0
            print(f"[gui] Batch export complete → {self.output_dir}")
            time.sleep(1.0)
            self._progress.visible = False
        finally:
            self._render_lock.release()

    # ==================================================================
    # Main loop
    # ==================================================================

    def run(self):
        """Block forever (call from ``if __name__ == '__main__'``)."""
        print("  Press Ctrl+C to quit.\n")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[gui] Shutting down.")


# ======================================================================
# Entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive camera trajectory editor for 3DGS depth+RGB rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ply", required=True, help="Path to the 3DGS .ply file")
    parser.add_argument("--config", default="config.yaml", help="config.yaml path")
    parser.add_argument("--binary", default=None, help="Path to vk_gaussian_splatting binary")
    parser.add_argument("--port", type=int, default=8080, help="Viser web server port")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    gui = TrajectoryGUI(
        ply_path=args.ply,
        config_path=args.config,
        binary_path=args.binary,
        port=args.port,
        output_dir=args.output,
    )
    gui.run()


if __name__ == "__main__":
    main()
