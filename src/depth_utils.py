"""
Depth post-processing utilities for 3DGS depth rendering.

Reads .hdr depth images from the Vulkan renderer's depth pass,
extracts float32 depth maps, and produces paired RGB+depth outputs.
"""

import os
import glob
import numpy as np

try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio as iio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_hdr_depth(hdr_path):
    """Load a .hdr depth image and extract the float32 depth map.
    
    The depth pass renders -viewZ into all 3 RGB channels. The R channel
    is extracted as the canonical depth (positive values = farther).
    
    Args:
        hdr_path: Path to .hdr file from depth rendering pass
        
    Returns:
        numpy array of shape (H, W) with float32 depth values
    """
    if HAS_CV2:
        # cv2 reads .hdr natively as float32 BGR
        img = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read HDR file: {hdr_path}")
        # All channels are identical; take the first (B in BGR = R in RGB for depth)
        depth = img[:, :, 0].astype(np.float32)
    elif HAS_IMAGEIO:
        img = iio.imread(hdr_path)
        depth = img[:, :, 0].astype(np.float32)
    else:
        raise ImportError("Neither cv2 nor imageio is available. Install one: pip install opencv-python imageio")
    
    return depth


def save_depth(depth, output_path, fmt="npy"):
    """Save a float32 depth map in the specified format.
    
    Args:
        depth: numpy array (H, W) float32
        output_path: Output file path (extension will be adjusted)
        fmt: Format - 'npy', 'exr', or 'pfm'
    """
    base = os.path.splitext(output_path)[0]
    
    if fmt == "npy":
        np.save(base + ".npy", depth)
    elif fmt == "exr":
        if not HAS_CV2:
            raise ImportError("OpenCV required for EXR output: pip install opencv-python")
        cv2.imwrite(base + ".exr", depth)
    elif fmt == "pfm":
        _write_pfm(base + ".pfm", depth)
    else:
        raise ValueError(f"Unknown depth format: {fmt}")


def depth_to_colormap(depth, colormap="viridis", min_val=None, max_val=None):
    """Convert a float32 depth map to a normalized colormap visualization.
    
    Args:
        depth: numpy array (H, W) float32
        colormap: Matplotlib colormap name or 'turbo' for cv2
        min_val: Minimum depth for normalization (default: auto)
        max_val: Maximum depth for normalization (default: auto)
        
    Returns:
        numpy array (H, W, 3) uint8 RGB colormap image
    """
    # Mask invalid pixels (zero depth = background)
    valid = depth > 0
    
    if min_val is None:
        min_val = depth[valid].min() if valid.any() else 0.0
    if max_val is None:
        max_val = depth[valid].max() if valid.any() else 1.0
    
    # Normalize to [0, 1]
    normalized = np.clip((depth - min_val) / (max_val - min_val + 1e-8), 0, 1)
    # Set background to 0
    normalized[~valid] = 0
    
    if HAS_CV2:
        # Use cv2 COLORMAP_TURBO (similar to viridis but built-in)
        gray = (normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        # Convert BGR to RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        # Set background to black
        colored[~valid] = 0
        return colored
    else:
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(colormap)
            colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
            colored[~valid] = 0
            return colored
        except ImportError:
            # Fallback: grayscale
            gray = (normalized * 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)


def process_depth_directory(hdr_dir, rgb_dir=None, output_dir=None, depth_format="npy"):
    """Process all .hdr depth files and create paired outputs.
    
    Args:
        hdr_dir: Directory containing .hdr depth files from depth pass
        rgb_dir: Directory containing .png RGB files from RGB pass (optional)
        output_dir: Output directory for paired results
        depth_format: Format for saving depth maps ('npy', 'exr', 'pfm')
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(hdr_dir), "paired")
    os.makedirs(output_dir, exist_ok=True)
    
    hdr_files = sorted(glob.glob(os.path.join(hdr_dir, "*.hdr")))
    if not hdr_files:
        print(f"No .hdr files found in {hdr_dir}")
        return
    
    print(f"Processing {len(hdr_files)} depth files...")
    
    # Collect all depths for global stats
    all_min = float('inf')
    all_max = float('-inf')
    
    for hdr_path in hdr_files:
        depth = load_hdr_depth(hdr_path)
        valid = depth > 0
        if valid.any():
            all_min = min(all_min, depth[valid].min())
            all_max = max(all_max, depth[valid].max())
    
    print(f"  Depth range: [{all_min:.4f}, {all_max:.4f}]")
    
    for hdr_path in hdr_files:
        basename = os.path.splitext(os.path.basename(hdr_path))[0]
        
        # Load depth
        depth = load_hdr_depth(hdr_path)
        
        # Validate
        if np.isnan(depth).any():
            print(f"  WARNING: {basename} contains NaN values")
        if np.isinf(depth).any():
            print(f"  WARNING: {basename} contains Inf values")
        
        # Save depth in requested format
        depth_out = os.path.join(output_dir, f"{basename}_depth")
        save_depth(depth, depth_out, fmt=depth_format)
        
        # Copy/link RGB if available
        if rgb_dir:
            rgb_path = os.path.join(rgb_dir, f"{basename}.png")
            if os.path.exists(rgb_path):
                rgb_out = os.path.join(output_dir, f"{basename}_rgb.png")
                if HAS_CV2:
                    rgb_img = cv2.imread(rgb_path)
                    cv2.imwrite(rgb_out, rgb_img)
                else:
                    import shutil
                    shutil.copy2(rgb_path, rgb_out)
        
        # Generate depth visualization
        vis = depth_to_colormap(depth, min_val=all_min, max_val=all_max)
        vis_path = os.path.join(output_dir, f"{basename}_depth_vis.png")
        if HAS_CV2:
            cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        elif HAS_IMAGEIO:
            iio.imwrite(vis_path, vis)
        
        print(f"  Processed: {basename}")
    
    print(f"Done. Paired outputs in {output_dir}")


def _write_pfm(path, data):
    """Write a PFM (Portable Float Map) file."""
    h, w = data.shape[:2]
    scale = -1.0  # little-endian
    
    with open(path, 'wb') as f:
        if data.ndim == 2:
            f.write(b'Pf\n')
        else:
            f.write(b'PF\n')
        f.write(f'{w} {h}\n'.encode())
        f.write(f'{scale}\n'.encode())
        # PFM stores rows bottom-to-top
        data = np.flipud(data).astype(np.float32)
        f.write(data.tobytes())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process depth HDR files from 3DGS renderer")
    parser.add_argument("--hdr-dir", required=True, help="Directory with .hdr depth files")
    parser.add_argument("--rgb-dir", default=None, help="Directory with .png RGB files")
    parser.add_argument("--output-dir", default=None, help="Output directory for paired results")
    parser.add_argument("--format", default="npy", choices=["npy", "exr", "pfm"],
                        help="Depth output format (default: npy)")
    
    args = parser.parse_args()
    
    process_depth_directory(
        hdr_dir=args.hdr_dir,
        rgb_dir=args.rgb_dir,
        output_dir=args.output_dir,
        depth_format=args.format
    )
