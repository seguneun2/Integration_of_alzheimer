from torch import nn
from torch.nn import functional as F
import torch
from .layers import *
from .encoder import GeneEncoder, MRIEncoder, MRIPatchEncoder
from .gene_embedder import GeneEmbedder
from .image_sampler import BasicDepthSampler2 as BasicDepthSampler , DeformablePatchSampler2d
from config import *

class Classifier(nn.Module):

    def __init__(self, in_channels=1, 
                       feature_dim=128, 
                       num_slices=10):

        super(Classifier, self).__init__()

        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.num_slices = num_slices

        self.slice_sampler = BasicDepthSampler(num_slices=num_slices)
        self.gene_embedder = GeneEmbedder(embedding_dim=feature_dim)
        self.mri_encoder = MRIEncoder(in_channels, feature_dim)
        self.gene_encoder = GeneEncoder(feature_dim)

        self.gene_classifier = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dim, feature_dim),

            nn.LeakyReLU(0.1, True),
            nn.Linear(feature_dim, 1),

            nn.Sigmoid()
        )

        self.gene_aggregator = nn.Linear(NUM_OF_GENES, 1)
        self.depth_aggregator = nn.Linear(num_slices, 1)

        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dim, feature_dim),

            nn.LeakyReLU(0.1, True),
            nn.Linear(feature_dim, 1),

            nn.Sigmoid()
        )

        self.mri_gene_attention = Attention(feature_dim)
        self.attention_mat  = None

    def encode_mri(self, mri):
        mri_local_feature, mri_global_feature = self.mri_encoder(mri)
        return mri_local_feature, mri_global_feature

    def encode_gene(self, gene):
        gene = self.gene_embedder(gene)
        gene_local_feature, gene_global_feature = self.gene_encoder(gene)
        return gene_local_feature, gene_global_feature

    def forward(self, mri, gene, return_features=False):
        N, C, D, H, W = mri.size()

        mri = self.slice_sampler(mri).permute(0, 2, 1, 3, 4)

        mri_local_feature, mri_global_feature = self.encode_mri(mri) 
        gene_local_feature, gene_global_feature = self.encode_gene(gene)

        mri_local_feature = mri_local_feature.permute(0, 2, 3, 1).reshape(N, -1, self.feature_dim) 
        
        query = gene_local_feature.detach() 
        key = mri_local_feature
        value = mri_local_feature

        local_feature, attention_mat  = self.mri_gene_attention(query, key, value)
        self.attention_mat = attention_mat
        local_feature = local_feature.transpose(1, 2)
        global_feature = self.gene_aggregator(local_feature).squeeze(-1)

        c = self.classifier(global_feature)

        if return_features:
            return c, mri_local_feature, mri_global_feature, gene_local_feature, gene_global_feature
        else:
            return c

    def compute_contrastive_loss(self,mri_global_feature, gene_global_feature):
        N = mri_global_feature.size(0)

        mri_global_feature = F.normalize(mri_global_feature, dim=-1, p=2, eps=1e-8)
        gene_global_feature = F.normalize(gene_global_feature, dim=-1, p=2, eps=1e-8)

        t = gene_global_feature.mm(mri_global_feature.t())
        t = -F.log_softmax(t, dim=-1)
        loss_contrast = torch.sum(t * torch.eye(N, device=t.device)) / N
        return loss_contrast

    def compute_gene_classification_loss(self, gene_local_feature, labels):
        preds = self.gene_classifier(gene_local_feature)
        labels = labels[:, None, None].expand_as(preds)
        loss_clf = F.binary_cross_entropy(preds, labels)
        return loss_clf

    def compute_classification_loss(self, preds, labels):
        preds = preds.view(-1)
        labels = labels.view(-1)
        loss_clf = F.binary_cross_entropy(preds, labels)
        return loss_clf
    



class Gene_Classifier(nn.Module):

    def __init__(self, in_channels=1, 
                       feature_dim=128, 
                       num_slices=10):

        super(Gene_Classifier, self).__init__()

        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.num_slices = num_slices

        self.slice_sampler = BasicDepthSampler(num_slices=num_slices)
        self.gene_embedder = GeneEmbedder(embedding_dim=feature_dim)
        self.gene_encoder = GeneEncoder(feature_dim)

        self.depth_aggregator = nn.Linear(num_slices, 1)

        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dim, feature_dim),

            nn.LeakyReLU(0.1, True),
            nn.Linear(feature_dim, 1),

            nn.Sigmoid()
        )


    def encode_gene(self, gene):
        gene = self.gene_embedder(gene)
        gene_local_feature, gene_global_feature = self.gene_encoder(gene)
        return gene_local_feature, gene_global_feature

    def forward(self, mri, gene, return_features=False):
        N, C, D, H, W = mri.size()

        mri = self.slice_sampler(mri).permute(0, 2, 1, 3, 4)

        gene_local_feature, gene_global_feature = self.encode_gene(gene)
        c = self.classifier(gene_global_feature)


        if return_features:
            return c, gene_local_feature, gene_global_feature
        else:
            return c

    def compute_classification_loss(self, preds, labels):
        preds = preds.view(-1)
        labels = labels.view(-1)
        loss_clf = F.binary_cross_entropy(preds, labels)
        return loss_clf
    
