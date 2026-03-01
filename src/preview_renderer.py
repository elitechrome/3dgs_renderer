"""Single-frame rendering wrapper for the Vulkan 3DGS binary.

Generates a temporary cameras.json and .cfg, runs the Vulkan binary
in headless mode, reads back the result, and returns numpy images.
"""

import os
import json
import hashlib
import tempfile
import subprocess
import numpy as np
from pathlib import Path


class PreviewRenderer:
    """Render individual frames on demand using the Vulkan binary."""

    def __init__(self, binary_path: str, ply_path: str, width: int, height: int):
        self.binary_path = os.path.abspath(binary_path)
        self.ply_path = os.path.abspath(ply_path)
        self.width = width
        self.height = height
        self._cache: dict[str, np.ndarray] = {}
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="3dgs_preview_"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_frame(self, camera_dict: dict, visualize: int = 0) -> np.ndarray | None:
        """Render a single frame and return the image as a numpy array.

        Args:
            camera_dict: INRIA-format camera dict (position, rotation, fx, fy, …).
            visualize: 0 = RGB, 2 = depth.

        Returns:
            For RGB (visualize=0): uint8 (H, W, 3) array, or *None* on error.
            For depth (visualize=2): float32 (H, W) array, or *None* on error.
        """
        cache_key = self._cache_key(camera_dict, visualize)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._run_binary(camera_dict, visualize)
        if result is not None:
            self._cache[cache_key] = result
        return result

    def render_rgb_and_depth(self, camera_dict: dict):
        """Convenience: render both RGB and depth for the same camera.

        Returns:
            (rgb, depth) where *rgb* is uint8 HWC and *depth* is float32 HW.
            Either may be *None* on failure.
        """
        rgb = self.render_frame(camera_dict, visualize=0)
        depth = self.render_frame(camera_dict, visualize=2)
        return rgb, depth

    def clear_cache(self):
        self._cache.clear()

    # ------------------------------------------------------------------
    # Batch rendering
    # ------------------------------------------------------------------

    def render_trajectory(
        self,
        camera_list: list[dict],
        visualize: int = 0,
        progress_callback=None,
    ) -> list[np.ndarray | None]:
        """Render a full list of cameras using a single binary invocation.

        This is much faster than calling :meth:`render_frame` in a loop
        because the Vulkan binary only starts up once.

        Args:
            camera_list: list of INRIA camera dicts.
            visualize: 0 = RGB, 2 = depth.
            progress_callback: optional ``fn(current, total)`` called after
                each frame is available.

        Returns:
            List of images (same length as *camera_list*).
        """
        n = len(camera_list)
        if n == 0:
            return []

        ext = ".hdr" if visualize == 2 else ".png"
        out_dir = self._tmp_dir / f"batch_{'depth' if visualize == 2 else 'rgb'}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write cameras.json
        cameras_json = out_dir / "cameras.json"
        serializable = []
        for i, cam in enumerate(camera_list):
            c = dict(cam)
            c["id"] = i
            c["img_name"] = f"frame_{i:05d}"
            for key in ("position", "rotation"):
                if isinstance(c.get(key), np.ndarray):
                    c[key] = c[key].tolist()
            serializable.append(c)
        cameras_json.write_text(json.dumps(serializable, indent=2))

        # Write .cfg
        cfg = self._generate_batch_cfg(cameras_json, out_dir, n, visualize, ext)

        # Run binary
        frame_count = n + 3  # sequences + headroom
        cmd = [
            self.binary_path,
            "--size", str(self.width), str(self.height),
            "--benchmark", "1",
            "--headless", "1",
            "--headless_frame_count", str(frame_count),
            "--visualize", str(visualize),
            "--sequencefile", str(cfg),
            self.ply_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"[PreviewRenderer] batch render failed: {exc}")
            return [None] * n

        # Read results
        results: list[np.ndarray | None] = []
        for i in range(n):
            path = out_dir / f"frame_{i:05d}{ext}"
            img = self._read_image(path, visualize)
            results.append(img)
            cache_key = self._cache_key(camera_list[i], visualize)
            if img is not None:
                self._cache[cache_key] = img
            if progress_callback:
                progress_callback(i + 1, n)
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_binary(self, camera_dict: dict, visualize: int) -> np.ndarray | None:
        """Run the Vulkan binary for a single camera."""
        ext = ".hdr" if visualize == 2 else ".png"
        tag = f"single_{visualize}"
        out_dir = self._tmp_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        # cameras.json
        cam = dict(camera_dict)
        cam["id"] = 0
        cam["img_name"] = "frame_00000"
        for key in ("position", "rotation"):
            if isinstance(cam.get(key), np.ndarray):
                cam[key] = cam[key].tolist()
        cameras_json = out_dir / "cameras.json"
        cameras_json.write_text(json.dumps([cam], indent=2))

        # .cfg
        cfg_path = out_dir / "render.cfg"
        screenshot_path = out_dir / f"frame_00000{ext}"
        cfg_content = (
            f'SEQUENCE "Init"\n'
            f'--visualize {visualize}\n'
            f'--cameraFile "{cameras_json.absolute()}"\n'
            f'--cameraIndex 0\n'
            f'--updateData 1\n\n'
            f'SEQUENCE "Capture"\n'
            f'--screenshot "{screenshot_path.absolute()}"\n\n'
        )
        cfg_path.write_text(cfg_content)

        cmd = [
            self.binary_path,
            "--size", str(self.width), str(self.height),
            "--benchmark", "1",
            "--headless", "1",
            "--headless_frame_count", "5",
            "--visualize", str(visualize),
            "--sequencefile", str(cfg_path),
            self.ply_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"[PreviewRenderer] render failed: {exc}")
            return None

        return self._read_image(screenshot_path, visualize)

    def _read_image(self, path: Path, visualize: int) -> np.ndarray | None:
        """Read an output image from disk."""
        if not path.exists():
            return None
        try:
            if visualize == 2:
                # HDR float depth
                try:
                    import cv2
                    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        return None
                    return img[:, :, 0].astype(np.float32)
                except ImportError:
                    import imageio.v3 as iio
                    img = iio.imread(str(path))
                    return img[:, :, 0].astype(np.float32)
            else:
                # PNG RGB
                try:
                    import cv2
                    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if img is None:
                        return None
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except ImportError:
                    import imageio.v3 as iio
                    return np.asarray(iio.imread(str(path)))
        except Exception as exc:
            print(f"[PreviewRenderer] failed to read {path}: {exc}")
            return None

    @staticmethod
    def _generate_batch_cfg(cameras_json, out_dir, num_frames, visualize, ext):
        """Generate a .cfg that captures *num_frames* screenshots."""
        lines = []
        lines.append(f'SEQUENCE "Init"')
        lines.append(f'--visualize {visualize}')
        lines.append(f'--cameraFile "{cameras_json.absolute()}"')
        lines.append(f'--cameraIndex 0')
        lines.append(f'--updateData 1')
        lines.append("")

        for i in range(num_frames):
            screenshot = out_dir / f"frame_{i:05d}{ext}"
            lines.append(f'SEQUENCE "Frame {i}"')
            lines.append(f'--screenshot "{screenshot.absolute()}"')
            if i + 1 < num_frames:
                lines.append(f'--cameraIndex {i + 1}')
            lines.append("")

        cfg_path = out_dir / f"render_{'depth' if visualize == 2 else 'rgb'}.cfg"
        cfg_path.write_text("\n".join(lines))
        return cfg_path

    @staticmethod
    def _cache_key(camera_dict: dict, visualize: int) -> str:
        """Deterministic hash of a camera + visualize mode."""
        blob = json.dumps(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in sorted(camera_dict.items())},
            sort_keys=True,
        ) + f"|vis={visualize}"
        return hashlib.md5(blob.encode()).hexdigest()
