import numpy as np
import math

class Camera:
    def __init__(self, R, T, FoVx, FoVy, width, height, znear=0.01, zfar=100.0):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height
        self.zfar = zfar
        self.znear = znear

        self.trans = T
        self.scale = 1.0

        self.world_view_transform = self.getWorld2View2()
        self.projection_matrix = self.getProjectionMatrix(znear, zfar, FoVx, FoVy)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def getWorld2View2(self):
        import torch
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = self.R.transpose()
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0
        return torch.tensor(Rt, dtype=torch.float32).cuda() # Assuming CUDA for now, or .to(device)

    def getProjectionMatrix(self, znear, zfar, fovX, fovY):
        import torch
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P.cuda() # Assuming CUDA

def fov_from_focal(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def get_camera_from_config(config, R, T):
    # R is 3x3 numpy, T is 3x1 numpy
    width = config['camera']['width']
    height = config['camera']['height']
    fx = config['camera']['fx']
    fy = config['camera']['fy']
    
    fovx = fov_from_focal(fx, width)
    fovy = fov_from_focal(fy, height)
    
    return Camera(R, T, fovx, fovy, width, height)
