
import os
import sys
import yaml
import numpy as np
import subprocess
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import io_utils 
import generator
import inria_io

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_cfg(cameras_json_path, output_dir, num_frames, visualize=0, ext=".png"):
    """Generate a .cfg sequence file for the Vulkan renderer.
    
    The screenshot callback fires at the START of a sequence (during prepareFrame),
    capturing the PREVIOUS frame's GBuffer. So we structure sequences as:
      - Init: load cameras, set camera 0, update data (renders camera 0)
      - For each frame i: screenshot (captures camera i from previous render), then set camera i+1
    
    Args:
        cameras_json_path: Path to cameras.json
        output_dir: Output directory for screenshots
        num_frames: Number of frames to render
        visualize: Visualization mode (0=final/RGB, 2=depth)
        ext: Screenshot file extension (.png or .hdr)
    
    Returns:
        Path to the generated .cfg file
    """
    cfg_content = ""
    
    # Sequence 0: Load scene, set visualize mode, load cameras, render camera 0
    cfg_content += f'SEQUENCE "Init"\n'
    cfg_content += f'--visualize {visualize}\n'
    cfg_content += f'--cameraFile "{cameras_json_path.absolute()}"\n'
    cfg_content += f'--cameraIndex 0\n'
    cfg_content += f'--updateData 1\n\n'
    
    # Each subsequent sequence: screenshot first (captures previous frame), then set next camera
    for i in range(num_frames):
        screenshot_path = output_dir / f"frame_{i:05d}{ext}"
        cfg_content += f'SEQUENCE "Frame {i}"\n'
        # Screenshot captures the GBuffer from the PREVIOUS frame's render (camera i)
        cfg_content += f'--screenshot "{screenshot_path.absolute()}"\n'
        # Set up the next camera for the next frame's render (unless last frame)
        if i + 1 < num_frames:
            cfg_content += f'--cameraIndex {i + 1}\n'
        cfg_content += '\n'

    suffix = "depth" if visualize == 2 else "rgb"
    benchmark_cfg_path = output_dir / f"render_{suffix}.cfg"
    with open(benchmark_cfg_path, 'w') as f:
        f.write(cfg_content)
    return benchmark_cfg_path, num_frames

def run_renderer(binary_path, ply_path, cfg_path, width, height, visualize=0, num_sequences=12):
    """Run the Vulkan renderer binary.
    
    Args:
        binary_path: Path to vk_gaussian_splatting executable
        ply_path: Path to input .ply file
        cfg_path: Path to .cfg sequence file
        width: Render width
        height: Render height
        visualize: Visualization mode (0=final/RGB, 2=depth)
        num_sequences: Total number of sequences in cfg (for headless frame count)
    """
    # headless_frame_count needs to be >= number of sequences + 1 (for completion detection)
    frame_count = num_sequences + 2
    cmd = [
        os.path.abspath(binary_path),
        "--size", str(width), str(height),
        "--benchmark", "1",
        "--headless", "1",
        "--headless_frame_count", str(frame_count),
        "--visualize", str(visualize),
        "--sequencefile", str(cfg_path.absolute()),
        os.path.abspath(ply_path)
    ]
    
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="3DGS Renderer using vk_gaussian_splatting backend")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--ply", type=str, required=True, help="Path to input .ply file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--binary", type=str, default=None, help="Path to vk_gaussian_splatting executable")
    parser.add_argument("--skip-rgb", action="store_true", help="Skip RGB rendering pass")
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth rendering pass")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Output setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    camera_cfg = config.get("camera", {})
    width = camera_cfg.get("width", 1920)
    height = camera_cfg.get("height", 1080)
    
    # Support both fov and fx/fy config styles
    if "fov" in camera_cfg:
        fov = np.deg2rad(camera_cfg.get("fov", 60))
    else:
        fx = camera_cfg.get("fx", width / 2)
        fov = 2 * np.arctan(width / (2 * fx))
    
    # Generate Poses
    print("Generating poses...")
    renderer_cfg = config.get("renderer", {})
    num_frames = renderer_cfg.get("num_frames", renderer_cfg.get("num_views", 10))
    radius = renderer_cfg.get("radius", 3.0)
    
    output_depth = renderer_cfg.get("output_depth", True)
    depth_format = renderer_cfg.get("depth_format", "npy")
    
    poses_data = generator.generate_uniform_poses(radius, num_frames)
    
    # Prepare INRIA cameras
    inria_cameras = []
    for i, (R, T, cam_pos) in enumerate(poses_data):
        view_matrix = np.eye(4)
        view_matrix[:3, :3] = R
        view_matrix[:3, 3] = T
        
        cam_dict = inria_io.view_matrix_to_inria(view_matrix, width, height, fov)
        cam_dict["id"] = i
        cam_dict["img_name"] = f"frame_{i:05d}"
        inria_cameras.append(cam_dict)
        
    # Save cameras.json
    cameras_json_path = output_dir / "cameras.json"
    inria_io.save_inria_cameras(inria_cameras, str(cameras_json_path))
    print(f"Saved cameras to {cameras_json_path}")
    
    # Find binary
    binary_path = args.binary
    if not binary_path:
        candidates = [
            # Release build has correct relative shader paths (../../shaders)
            "vk_gaussian_splatting/_bin/Release/vk_gaussian_splatting",
            "vk_gaussian_splatting/build/vk_gaussian_splatting",
            "vk_gaussian_splatting/build/bin/vk_gaussian_splatting",
            "vk_gaussian_splatting/_bin/vk_gaussian_splatting",
        ]
        for c in candidates:
            if os.path.exists(c):
                binary_path = c
                break
                
    if not binary_path or not os.path.exists(binary_path):
        print("\n[IMPORTANT] vk_gaussian_splatting binary not found.")
        print("Please build the backend first. See workplan for build instructions.")
        print(f"Once built, run: python main.py --ply {args.ply} --binary <path_to_executable>")
        return

    # macOS Vulkan Fix
    if sys.platform == "darwin":
        if "VK_ICD_FILENAMES" not in os.environ:
            candidates = [
                "/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json",
                "/usr/local/share/vulkan/icd.d/MoltenVK_icd.json",
                "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json",
                "/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json",
            ]
            for c in candidates:
                if os.path.exists(c):
                    os.environ["VK_ICD_FILENAMES"] = c
                    print(f"[macOS] Set VK_ICD_FILENAMES to {c}")
                    break

    # Pass 1: RGB rendering
    if not args.skip_rgb:
        print("\n=== Pass 1: RGB Rendering ===")
        rgb_cfg, rgb_num_seq = generate_cfg(cameras_json_path, rgb_dir, num_frames, visualize=0, ext=".png")
        print(f"Saved RGB config to {rgb_cfg}")
        try:
            run_renderer(binary_path, args.ply, rgb_cfg, width, height, visualize=0, num_sequences=rgb_num_seq + 1)
            print("RGB rendering complete!")
        except subprocess.CalledProcessError as e:
            print(f"Error during RGB rendering: {e}")
            return

    # Pass 2: Depth rendering
    if not args.skip_depth and output_depth:
        print("\n=== Pass 2: Depth Rendering ===")
        depth_cfg, depth_num_seq = generate_cfg(cameras_json_path, depth_dir, num_frames, visualize=2, ext=".hdr")
        print(f"Saved depth config to {depth_cfg}")
        try:
            run_renderer(binary_path, args.ply, depth_cfg, width, height, visualize=2, num_sequences=depth_num_seq + 1)
            print("Depth rendering complete!")
        except subprocess.CalledProcessError as e:
            print(f"Error during depth rendering: {e}")
            return

    # Post-processing: convert .hdr depth to numpy arrays
    if not args.skip_depth and output_depth:
        print("\n=== Post-processing depth maps ===")
        try:
            from depth_utils import process_depth_directory
            process_depth_directory(
                hdr_dir=str(depth_dir),
                rgb_dir=str(rgb_dir),
                output_dir=str(output_dir / "paired"),
                depth_format=depth_format
            )
            print(f"Paired outputs saved to {output_dir / 'paired'}")
        except ImportError:
            print("Warning: depth_utils not found. Skipping post-processing.")
            print("Raw .hdr depth files are available in:", depth_dir)
        except Exception as e:
            print(f"Error during post-processing: {e}")
            print("Raw .hdr depth files are available in:", depth_dir)

    print("\nAll rendering complete!")
    print(f"  RGB images:   {rgb_dir}")
    if output_depth:
        print(f"  Depth images: {depth_dir}")
        print(f"  Paired data:  {output_dir / 'paired'}")

if __name__ == "__main__":
    main()
