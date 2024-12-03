import torch
from torch import nn
from torch.nn import functional as F


class ResidualLayer(nn.Module):

    def __init__(self, feature_dim):
        super(ResidualLayer, self).__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feature_dim, feature_dim, 3, stride=1, padding=1),

            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feature_dim, feature_dim, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.residual(x) + x


class MaskedResidualLayer(nn.Module):

    def __init__(self, feature_dim):
        super(MaskedResidualLayer, self).__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feature_dim, feature_dim, 3, stride=1, padding=1),

            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feature_dim, feature_dim, 3, stride=1, padding=1),
        )

        self.mask_predictor = nn.Sequential(
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feature_dim, feature_dim, 3, stride=1, padding=1),

            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feature_dim, feature_dim, 3, stride=1, padding=1),

            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.mask_predictor(x)
        return (self.residual(x) + x) * mask



class ResidualLayer2d(nn.Module):

    def __init__(self, feature_dim):
        super(ResidualLayer2d, self).__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feature_dim, feature_dim, 3, stride=1, padding=1),

            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feature_dim, feature_dim, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.residual(x) + x


class Attention(nn.Module):

    def __init__(self, feature_dim=128):
        super(Attention, self).__init__()

        self.query_embedder = nn.Linear(feature_dim, feature_dim)
        self.key_embedder = nn.Linear(feature_dim, feature_dim)

    def forward(self, query, key, value):
        query = self.query_embedder(query)
        key = self.key_embedder(key)
        value = self.key_embedder(value)

        attention_mat = query.bmm(key.transpose(1, 2))
        attention_mat = F.softmax(attention_mat, dim=-1)

        result = attention_mat.bmm(value)
        return result, attention_mat
