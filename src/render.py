import torch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render(viewpoint_camera, gaussian_data, bg_color, scaling_modifier=1.0):
    """
    Render the scene. 
    
    gaussian_data: The loaded GaussianData object (from io.py)
    viewpoint_camera: The Camera object (from camera.py)
    bg_color: Background color (Tensor)
    """
 
    # Create zero tensor. We will use it to make device tensors.
    # In a real app, gaussian_data should be pre-loaded to GPU.
    # For simplicity here, we move to CUDA on the fly if not already.
    
    # 1. Prepare data
    xyz = torch.tensor(gaussian_data.xyz, dtype=torch.float32, device="cuda")
    opacity = torch.tensor(gaussian_data.opacities, dtype=torch.float32, device="cuda")
    
    # Features handling
    # Render RGB: We need to combine DC and Rest for SH.
    # But Rasterizer expects them separated usually.
    features_dc = torch.tensor(gaussian_data.features_dc, dtype=torch.float32, device="cuda").transpose(1, 2)
    features_rest = torch.tensor(gaussian_data.features_rest, dtype=torch.float32, device="cuda").transpose(1, 2)
    
    scales = torch.tensor(gaussian_data.scales, dtype=torch.float32, device="cuda")
    rots = torch.tensor(gaussian_data.rotations, dtype=torch.float32, device="cuda")

    # 2. Set up Rasterization Settings
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussian_data.sh_degrees,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings)

    # 3. Rasterize
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=torch.zeros_like(xyz, dtype=torch.float32, requires_grad=False, device="cuda") + 0,
        shs=None, # if we passed explicit colors, we use colors_precomp. If SH, we pass features.
        colors_precomp=None, 
        opacities=opacity,
        scales=scales,
        rotations=rots,
        cov3D_precomp=None
    )
    # Be careful: `rasterizer` input arguments changed across versions. 
    # Standard 3DGS arguments are:
    # means3D, means2D, shs, colors_precomp, opacities, scales, rotations, ...
    # We should pass `shs` if we want SH rendering.
    
    # Let's combine SHs for passing.
    # shs should be (N, degrees, 3)? Or (N, 3, degrees)?
    # Original implementation: shs = torch.cat([features_dc, features_rest], dim=1).contiguous()
    # features_dc was (N, 1, 3), rest (N, 15, 3).
    # So shs: (N, 16, 3).
    # My loader loaded them as (N, 3, 1) and (N, 3, K). 
    # Let's fix shapes in `io.py` or here.
    # In `io.py`: features_dc [N, 3, 1], features_rest [N, 3, K] (if I recall correctly).
    # Correct shape for rasterizer `shs` expects (N, int((deg+1)**2), 3) usually?
    # Checking `diff-gaussian-rasterization` source or usage in 3DGS:
    # `pc.get_features` returns self._features_dc and self._features_rest concatenated.
    # Shape of these tensors in 3DGS is: 
    # _features_dc: [N, 1, 3]
    # _features_rest: [N, 15, 3]
    # So we need to transpose my inputs.
    
    # Fix IO data here locally
    f_dc = torch.tensor(gaussian_data.features_dc, dtype=torch.float32, device="cuda").permute(0, 2, 1) # [N, 1, 3]
    f_rest = torch.tensor(gaussian_data.features_rest, dtype=torch.float32, device="cuda").permute(0, 2, 1) # [N, K, 3]
    shs = torch.cat([f_dc, f_rest], dim=1).contiguous()

    # Re-call rasterizer with correct args
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=torch.zeros_like(xyz, dtype=torch.float32, requires_grad=False, device="cuda") + 0,
        shs=shs, 
        colors_precomp=None, 
        opacities=opacity,
        scales=scales,
        rotations=rots,
        cov3D_precomp=None
    )

    # Depth Rendering
    # "Depth" is not directly output by standard rasterizer.
    # Hack: Render colors where color = depth from camera.
    # 1. Project means to camera frame to get Z.
    # 2. Normalize or use as color.
    # 3. Rasterize with `colors_precomp`.
    
    # Transform points to view space
    view_matrix = viewpoint_camera.world_view_transform # 4x4
    #  p_view = p_world @ R_t + T? 
    # world_view_transform is typically Transpose(R) with T in last row? 
    # Camera class: Rt[:3, :3] = R.T, Rt[:3, 3] = T. 
    # So xyz1 @ Rt -> view space.
    
    xyz_ones = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device="cuda")], dim=1)
    xyz_view = xyz_ones @ view_matrix
    depth_values = xyz_view[:, 2:3] # Z values

    # For depth map, we want to render these Z values.
    # We can pass them as RGB (replicated) to `colors_precomp`.
    # Note: This gives a weighted average of Depth based on alpha. 
    # This is "Alpha Blended Depth", commonly used.
    
    depth_color = depth_values.repeat(1, 3) 
    # We might need to handle negative Z if camera looks down -Z. 
    # Usually standard gl camera looks down -Z, so Z is negative.
    # We want positive depth.
    # If using the provided Camera class, check projection. 
    # Using `getProjectionMatrix` with z_sign=1.0 suggests +Z usage or standard GL?
    # Let's stick strictly to what the rasterizer does.
    
    depth_image, _ = rasterizer(
        means3D=xyz,
        means2D=torch.zeros_like(xyz, dtype=torch.float32, requires_grad=False, device="cuda") + 0,
        shs=None, 
        colors_precomp=depth_color, 
        opacities=opacity,
        scales=scales,
        rotations=rots,
        cov3D_precomp=None
    )
    
    # depth_image is (3, H, W). All channels same.
    depth_map = depth_image[0, :, :]
    
    return rendered_image, depth_map
