
import json
import numpy as np

def save_inria_cameras(cameras, filename):
    """
    Saves a list of camera parameters to a JSON file in INRIA format.
    
    Args:
        cameras: List of dictionaries, each containing:
            'id': int
            'img_name': str
            'width': int
            'height': int
            'position': list/array [x, y, z]
            'rotation': list of lists [[r00, r01, r02], ...] (3x3 matrix)
            'fx': float
            'fy': float
        filename: Output filename
    """
    
    # Ensure numpy arrays are converted to lists
    serializable_cameras = []
    for cam in cameras:
        cam_dict = cam.copy()
        if isinstance(cam_dict['position'], np.ndarray):
            cam_dict['position'] = cam_dict['position'].tolist()
        if isinstance(cam_dict['rotation'], np.ndarray):
            cam_dict['rotation'] = cam_dict['rotation'].tolist()
        serializable_cameras.append(cam_dict)
        
    with open(filename, 'w') as f:
        json.dump(serializable_cameras, f, indent=4)

def view_matrix_to_inria(view_matrix, width, height, fov_x):
    """
    Converts a 4x4 View Matrix (World-to-Camera) to INRIA format components.
    INRIA format expects:
    - position: Camera eye position (from C2W)
    - rotation: Rotation matrix (C2W top-left 3x3)
    - fx, fy: Focal lengths derived from FOV
    
    Args:
        view_matrix: 4x4 numpy array (W2C)
        width: int
        height: int
        fov_x: float (radians)
        
    Returns:
        dict with 'position', 'rotation', 'fx', 'fy', 'width', 'height'
    """
    # Invert View Matrix to get Camera-to-World
    c2w = np.linalg.inv(view_matrix)
    
    position = c2w[:3, 3]
    rotation = c2w[:3, :3]
    
    # Calculate focal length
    # fov_x = 2 * arctan(w / (2 * fx))
    # fx = w / (2 * tan(fov_x / 2))
    fx = width / (2 * np.tan(fov_x / 2))
    
    # Assuming square pixels, fy = fx
    # Or derive from fov_y if available. 
    # For now assume square pixels.
    fy = fx
    
    return {
        'width': width,
        'height': height,
        'position': position,
        'rotation': rotation,
        'fx': fx,
        'fy': fy
    }
