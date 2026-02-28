import yaml
import numpy as np
from plyfile import PlyData, PlyElement
import torch

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class GaussianData:
    def __init__(self, xyz, opacities, features_dc, features_rest, scales, rotations):
        self.xyz = xyz
        self.opacities = opacities
        self.features_dc = features_dc
        self.features_rest = features_rest
        self.scales = scales
        self.rotations = rotations

    @property
    def sh_degrees(self):
        # Infer SH degree from feature count
        # (degrees + 1)^2 * 3 = total_features (if 3 channels)
        # But features_rest usually has (n_features - 3) channels.
        # DC is always degree 0 (1 coeff per channel).
        return int((self.features_rest.shape[1] / 3 + 1) ** 0.5) - 1

def load_ply(path):
    plydata = PlyData.read(path)
    
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    
    if len(extra_f_names) > 0:
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape to (N, 3, (degrees+1)^2 - 1) logic roughly
        # Standard implementation splits standard channels.
        # We will keep it flattened or standard tensor format as needed by renderer.
        # Usually standard rasterizer expects (N, features, 3) or (N, 3, features)?
        # Original 3DGS implementation expects features_dc [N, 1, 3] and features_rest [N, 15, 3] usually.
        # Wait, original is [N, 3] and [N, 45] (flattened)
        # Let's keep as numpy for now, render.py will convert to torch and reshape.
        features_rest = features_extra.reshape((xyz.shape[0], 3, -1)) # Shape (N, 3, K)
    else:
        features_rest = np.zeros((xyz.shape[0], 3, 0))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rotations = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rotations[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return GaussianData(xyz, opacities, features_dc, features_rest, scales, rotations)
