# Depth + RGB Pairs from 3D Gaussian Splatting

Generate paired RGB and float32 depth images from 3DGS `.ply` scenes using a Vulkan-based renderer. Includes an interactive web GUI for designing camera trajectories and a headless batch pipeline.

## Overview

This project wraps NVIDIA's [vk_gaussian_splatting](https://github.com/nvpro-samples/vk_gaussian_splatting) Vulkan renderer with Python tooling to produce aligned RGB + depth pairs from trained 3D Gaussian Splatting scenes.

**How depth works:** Standard 3DGS rendering disables hardware depth writes (alpha-blended splats), leaving the Z-buffer empty. Instead, we render depth via a two-pass approach — the second pass outputs each Gaussian's view-space Z through the same alpha-blending pipeline, producing per-pixel alpha-composited depth stored in float32 HDR format.

### Features

- **Two-pass rendering** — RGB (`.png`) + depth (`.hdr`) with identical camera parameters
- **Interactive GUI** — Viser-based web viewer with Gaussian splat visualization, keyframe editing, timeline playback, and live RGB/depth preview
- **Batch pipeline** — Headless rendering of arbitrary camera trajectories
- **Depth post-processing** — HDR → NumPy/EXR/PFM conversion with colormapped visualization

## Project Structure

```
3dgs_renderer/
├── gui.py                  # Interactive camera trajectory editor (Viser web GUI)
├── main.py                 # Headless batch rendering pipeline
├── config.yaml             # Camera & renderer settings
├── requirements.txt        # Python dependencies
├── setup_uv.sh             # Environment setup via uv
├── run.sh                  # Quick-start script
├── src/
│   ├── trajectory.py       # Keyframe storage + cubic-spline/Slerp interpolation
│   ├── preview_renderer.py # Single-frame Vulkan binary wrapper
│   ├── depth_utils.py      # HDR depth → numpy post-processing
│   ├── inria_io.py         # INRIA camera JSON I/O
│   ├── generator.py        # Uniform camera pose generation
│   ├── camera.py           # Camera model utilities
│   └── io_utils.py         # File I/O helpers
└── vk_gaussian_splatting/  # Vulkan renderer (git submodule)
    ├── _bin/               # Built binary
    └── _downloaded_resources/
        └── flowers_1/      # Sample scene
```

## Prerequisites

- **Linux** with a Vulkan-capable GPU (NVIDIA recommended)
- **Vulkan SDK** 1.4.x (bundled in `1.4.321.0/`)
- **CMake** ≥ 3.20 and a C++17 compiler
- **Python** 3.10+
- **uv** (recommended) or pip

## Setup

### 1. Build the Vulkan renderer

```bash
cd 3dgs_renderer
source 1.4.321.0/setup-env.sh
cd vk_gaussian_splatting/build
cmake ..
make -j$(nproc)
cd ../..
```

The binary is placed at `vk_gaussian_splatting/_bin/Release/vk_gaussian_splatting`.

### 2. Install Python dependencies

```bash
# Using uv (recommended)
bash setup_uv.sh

# Or manually
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Interactive GUI

Launch the web-based trajectory editor:

```bash
source .venv/bin/activate
python gui.py --ply vk_gaussian_splatting/_downloaded_resources/flowers_1/flowers_1.ply
```

Open `http://localhost:8080` in your browser. The GUI provides:

- **3D viewport** — Navigate the scene rendered as Gaussian splats
- **Keyframe capture** — Press "Add Keyframe" to save the current camera pose
- **Timeline** — Scrub through interpolated frames with play/pause controls
- **Live preview** — Render RGB + depth at any frame via the Vulkan backend
- **Batch export** — Render all frames and export `cameras.json`

Options:
```
--ply PATH       Path to 3DGS .ply scene file (required)
--config PATH    Config file (default: config.yaml)
--binary PATH    Vulkan binary override
--port PORT      Web server port (default: 8080)
--output DIR     Output directory (default: output/)
```

### Headless Batch Rendering

Generate uniform camera poses and render all frames:

```bash
source .venv/bin/activate
python main.py --ply vk_gaussian_splatting/_downloaded_resources/flowers_1/flowers_1.ply
```

Or use the quick-start script:

```bash
bash run.sh
```

Options:
```
--ply PATH       Path to 3DGS .ply scene file (required)
--config PATH    Config file (default: config.yaml)
--binary PATH    Vulkan binary override
--output DIR     Output directory (default: output/)
--skip-rgb       Skip RGB rendering pass
--skip-depth     Skip depth rendering pass
```

### Output Structure

```
output/
├── cameras.json            # INRIA-format camera parameters
├── rgb/
│   ├── frame_00000.png     # RGB images
│   └── ...
├── depth/
│   ├── frame_00000.hdr     # Float32 depth (Radiance HDR)
│   └── ...
└── paired/
    ├── frame_00000_rgb.png
    ├── frame_00000_depth.npy   # Float32 depth as NumPy array
    ├── frame_00000_depth_vis.png  # Colormapped depth visualization
    └── ...
```

## Configuration

Edit `config.yaml`:

```yaml
camera:
  width: 1024
  height: 768
  fx: 800.0          # Focal length in pixels
  fy: 800.0

renderer:
  radius: 3.0        # Camera orbit radius
  num_frames: 10     # Number of views to generate
  output_depth: true  # Enable depth pass
  depth_format: npy   # Output format: npy, exr, or pfm
```

## How It Works

### Two-Pass Rendering

1. **RGB pass** — The Vulkan binary renders normally (`--visualize 0`), saving `.png` screenshots
2. **Depth pass** — The same binary re-renders with `--visualize 2`, where the fragment shader outputs `-viewZ` (negated camera-space Z) as the "color" through the same alpha-blending pipeline. A `R32G32B32A32_SFLOAT` framebuffer preserves full float32 precision. Screenshots are saved as `.hdr` files.

The alpha-blending equation ($\alpha_i \cdot d_i + (1 - \alpha_i) \cdot d_{acc}$) composites depth identically to color, producing correct alpha-weighted depth per pixel that accounts for all overlapping semi-transparent Gaussians.

### GUI Architecture

The GUI uses [Viser](https://viser.studio/) for the web-based 3D viewer:
- Gaussian splats are loaded from the PLY file and displayed via `scene.add_gaussian_splats()`
- Camera keyframes are interpolated with cubic splines (position) and Slerp (orientation)
- The Vulkan binary is invoked on-demand for high-quality RGB/depth previews
- Exported cameras use the INRIA JSON format consumed by `vk_gaussian_splatting`
