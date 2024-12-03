import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.layers import *
from config import *


class BasicDepthSampler(nn.Module):
    def __init__(self, num_slices):
        super(BasicDepthSampler, self).__init__()

        self.num_slices = num_slices

    def forward(self, x):
        D = x.size(2)
        return x[:, :, ::int(np.ceil(D/self.num_slices))]


class BasicDepthSampler2(nn.Module):
    def __init__(self, num_slices=10):
        super(BasicDepthSampler2, self).__init__()

        self.num_slices = num_slices

    def forward(self, x):
        H = x.size(-1)//2
        return x[:, :, H-self.num_slices//2: H+self.num_slices//2]


class DeformablePatchSampler2d(nn.Module):

    def __init__(self, patch_size=16, 
                       num_patch_height=4, 
                       num_patch_width=4):
                 
        super(DeformablePatchSampler2d, self).__init__()

        self.patch_size = patch_size
        self.num_patch_height = num_patch_height
        self.num_patch_width = num_patch_width

        # self.offset_predictor = OffsetPredictor(feature_dim)
        # self.embedder = nn.Conv3d(in_features, feature_dim, 1, stride=1, padding=0)
        self.offset = nn.Parameter(
            torch.randn(num_patch_height*num_patch_width, 2) * 0.001,
            requires_grad=True
        )

    def forward(self, x):
        """
        Arguments:
        ----------
        - x: (N, C, H, W)

        Returns:
        --------
        - patches: (N, M, C, patch_size, patch_size)
        """
        N = x.size(0)

        patch_coords, patch_center_coords = self.sample_coords_center(x)
        patch_coords_deformed = patch_coords + self.offset[None, :, None, None, :]
        patches = list(map(lambda i: self.sample_patches(x[i], patch_coords_deformed[i]), range(N)))
        patches = torch.stack(patches, dim=0)
        return patches

    def compute_patch_coords(self, center_coord):
        """
        Arguments:
        ----------
        - patch_coord: (3,)

        Returns:
        --------
        - patch_coords: (patch_size, patch_size, 2)
        """

        H, W = center_coord

        h_coords = torch.arange(H - self.patch_size//2, H + self.patch_size//2, dtype=torch.float32)
        w_coords = torch.arange(W - self.patch_size//2, W + self.patch_size//2, dtype=torch.float32)

        patch_coords = torch.cartesian_prod(h_coords, w_coords)
        patch_coords = patch_coords.view(self.patch_size, self.patch_size, 2)
        return patch_coords

    def sample_coords_center(self, x):
        """
        Arguments:
        ----------
        - x: (N, C, D, H, W)

        Returns:
        --------
        - patch_coords: (N, M, patch_size, patch_size, 2)
        - patch_center_coords: (N, M, 3)
        """
        N, C, H, W = x.size()
        M = self.num_patch_height * self.num_patch_width

        patch_coords_H = torch.linspace(0, H, steps=self.num_patch_height + 4, dtype=torch.float32)[2:-2]
        patch_coords_W = torch.linspace(0, W, steps=self.num_patch_width + 4, dtype=torch.float32)[2:-2]

        patch_center_coords = torch.cartesian_prod(patch_coords_H, patch_coords_W)
        patch_coords = list(map(lambda center_coord: self.compute_patch_coords(center_coord), patch_center_coords))
        patch_coords = torch.stack(patch_coords, dim=0)

        patch_coords = patch_coords[None, :, :, :, :].expand(N, M, self.patch_size, self.patch_size, 2)
        patch_center_coords = patch_center_coords[None, :, :].expand(N, M, 2)

        patch_coords = patch_coords.to(x.device)
        patch_center_coords = patch_center_coords.to(x.device)

        patch_coords[..., 0] = 2*(patch_coords[..., 0]/(H - 1)) - 1
        patch_coords[..., 1] = 2*(patch_coords[..., 1]/(W - 1)) - 1

        patch_center_coords[..., 0] = 2*(patch_center_coords[..., 0]/(H - 1)) - 1
        patch_center_coords[..., 1] = 2*(patch_center_coords[..., 1]/(W - 1)) - 1

        return patch_coords, patch_center_coords

    def sample_patches(self, x, patch_coords):
        """
        Arguments:
        ----------
        - x: (C, H, W)
        - patch_coords: (M, patch_size, patch_size, 3)

        Returns:
        --------
        - patches: (M, C, H, W)
        """
        
        C, H, W = x.size()
        M = patch_coords.size(0)

        x = x[None, :, :, :].expand(M, C, H, W)
        patches = F.grid_sample(x, patch_coords, align_corners=False)
        return patches

