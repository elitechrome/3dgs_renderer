import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def look_at(cam_pos, target):
    # Returns R, T compliant with the Camera class expectations
    # Camera class expects R and T such that Point_cam = R * Point_world + T
    # Usually look_at produces a World-to-Camera matrix directly.
    
    forward = normalize(target - cam_pos) # +Z direction usually for OpenGL, but check convention
    # In standard 3DGS (and OpenGL), camera looks down -Z? No, +Z is backwards.
    # We want camera to look at target. 
    # Convention in diff-gaussian-rasterization:
    # "The rasterizer expects the View Matrix to transform from World Space to View Space."
    # Standard decomposition R, T.
    
    # Vectors for the camera coordinate system usually: 
    # Right: +X, Up: +Y, Forward: -Z (looking into screen) or +Z (looking away).
    # Creating a basis:
    # Z (forward) = normalize(cam_pos - target) (Vector pointing to camera from target)
    z_axis = normalize(cam_pos - target) 
    
    # We need an up vector. Global Up is usually (0,1,0).
    up = np.array([0.0, 1.0, 0.0])
    
    # If looking straight down/up, handle singularity
    if np.abs(np.dot(z_axis, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
        
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    
    # Rotation Matrix (World to Camera)
    # Rows are the basis vectors of the camera in world coords? 
    # No, R * P yields projection on axis. 
    # R = [Xx, Xy, Xz; Yx, Yy, Yz; Zx, Zy, Zz]
    R = np.vstack([x_axis, y_axis, z_axis])
    
    # T puts the camera at origin. T = -R * C
    T = -R @ cam_pos
    
    return R.T, T # Helper returns R expected by our Camera class (often Transpose of View Matrix rotation part if using standard convention, need to Verify)
    # Re-verifying Camera class usage:
    # Rt[:3, :3] = self.R.transpose()
    # So if we pass 'R' here which is World2Cam rotation, then Camera class transposes it? 
    # That implies Camera class stores Cam2World rotation? 
    # Let's check typical 3DGS code. 
    # Typical: self.world_view_transform constructed by transposing R and setting T. 
    # If self.R is passed as input.
    # If input R is World2View Rotation, and we transpose it, we get View2World.
    # BUT the ViewMatrix construction usually is:
    # [[R_row0, Tx], [R_row1, Ty], ...] 
    # The snippet used: Rt[:3, :3] = self.R.transpose()
    # This suggests self.R is assumed to be Column-Major or the code expects Col-Major loading?
    # PyTorch3D / OpenGL usually Column Major in logic but Row Major memory. 
    # Let's stick to standard LookAt WorldToView R.
    # If Camera class does R.transpose(), it might be converting Row-Major R to Column-Major for torch/gl logic.
    # I will return standard View Matrix R and T.
    # R: 3x3, T: 3 vectors.

    # Re-reading Camera class:
    # Rt[:3, :3] = self.R.transpose()...
    # Usually this means self.R is expected to be [Right, Up, Back] as columns?
    # Or it's coping with a specific codebase idiosyncrasy.
    # I will assume standard R (rows are bases) and T.
    return R, T

def generate_uniform_poses(radius, count, center=np.array([0,0,0])):
    poses = []
    # Fibonacci sphere for uniform distribution on sphere
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    
    for i in range(count):
        y = 1 - (i / float(count - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y) 
        
        theta = phi * i 
        
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        
        cam_pos = np.array([x, y, z]) * radius + center
        
        R, T = look_at(cam_pos, center)
        poses.append((R, T, cam_pos))
        
    return poses

if __name__ == "__main__":
    # Test
    poses = generate_uniform_poses(3.0, 10)
    print(poses[0])
