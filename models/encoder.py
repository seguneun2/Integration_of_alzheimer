import torch
from torch import nn
from .layers import *


class MRIEncoder(nn.Module):

    def __init__(self, in_features=1, feature_dim=128):
        super(MRIEncoder, self).__init__()

        self.feature_dim = feature_dim

        self.down_sampling = nn.Sequential(
            nn.Conv2d(in_features, feature_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(feature_dim, feature_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, True),
            
            nn.Conv2d(feature_dim, feature_dim, 3, stride=2, padding=1),
        )

        self.local_feature_encoder = nn.Sequential(
            MaskedResidualLayer(feature_dim),
            MaskedResidualLayer(feature_dim),
            MaskedResidualLayer(feature_dim),
            MaskedResidualLayer(feature_dim),
            MaskedResidualLayer(feature_dim),
        )

        self.global_feature_encoder = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dim, feature_dim),

            nn.LeakyReLU(0.1, True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x):
        """
        Arguments:
        - patches: (N*num_slices, 1, patch_size, patch_size)
        """

        N, S, C, H, W = x.size()
        x = x.reshape(N*S, C, H, W)

        f = self.down_sampling(x)
        local_feature = self.local_feature_encoder(f)
        local_feature = local_feature.view(N, S, self.feature_dim, -1)

        global_feature = local_feature.mean(dim=(1, 3))
        global_feature = self.global_feature_encoder(global_feature)
        return local_feature, global_feature


class MRIPatchEncoder(nn.Module):

    def __init__(self, in_features=1, feature_dim=128, patch_size=16):
        super(MRIPatchEncoder, self).__init__()

        self.feature_dim = feature_dim

        self.down_sampling = nn.Sequential(
            nn.Conv2d(in_features, feature_dim, patch_size, stride=1, padding=0),
        )

        self.local_feature_encoder = nn.Sequential(
            nn.TransformerEncoderLayer(feature_dim, 8),# batch_first=True),
            nn.TransformerEncoderLayer(feature_dim, 8), #batch_first=True),
            nn.TransformerEncoderLayer(feature_dim, 8), #batch_first=True),
            nn.TransformerEncoderLayer(feature_dim, 8), #batch_first=True),
            nn.TransformerEncoderLayer(feature_dim, 8), #batch_first=True),
        )
    

        self.global_feature_encoder = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dim, feature_dim),

            nn.LeakyReLU(0.1, True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, patches):
        """
        Arguments:
        - patches: (N, num_slice, num_patches, 1, patch_size, patch_size)
        """

        N, S, P, C, H, W = patches.size()
        patches = patches.view(N*S*P, C, H, W)

        f = self.down_sampling(patches).view(N, S*P, self.feature_dim)

        local_feature = self.local_feature_encoder(f)
    
        global_feature = local_feature.mean(dim=1)
        global_feature = self.global_feature_encoder(global_feature)
        return local_feature, global_feature



    
    
class GeneEncoder(nn.Module):

    def __init__(self, feature_dim=128):
        super(GeneEncoder, self).__init__()

        self.local_feature_encoder = nn.Sequential(
            nn.TransformerEncoderLayer(feature_dim, 8),# batch_first=True),
            nn.TransformerEncoderLayer(feature_dim, 8), #batch_first=True),
            nn.TransformerEncoderLayer(feature_dim, 8), #batch_first=True),
            nn.TransformerEncoderLayer(feature_dim, 8), #batch_first=True),
            nn.TransformerEncoderLayer(feature_dim, 8), #batch_first=True),
        )
        self.encode_layer = nn.Sequential(nn.Linear(feature_dim, feature_dim//2),  
                                          nn.LeakyReLU(True),
                                          nn.Linear(feature_dim//2, feature_dim))
        
        
        self.global_feature_encoder = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dim, feature_dim),

            nn.LeakyReLU(0.1, True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, gene_seq_embedding):
        """
        Arguments:
        ----------
        - gene_seq: (N, len_of_seq, embedding_dim)
        """
        # 1.8.0 batch first not supported
        f = gene_seq_embedding
        local_feature = self.encode_layer(f)
        # local_feature = self.local_feature_encoder(f)
        global_feature = local_feature.mean(dim=1)
        global_feature = self.global_feature_encoder(global_feature)
        return local_feature, global_feature


class LatentEncoder(nn.Module):
    ## Linear --> Linear

    def __init__(self, feature_dim):
        super(LatentEncoder, self).__init__()

        self.embedder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(True),

            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x):
        return self.embedder(x)

