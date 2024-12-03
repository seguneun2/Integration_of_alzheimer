import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from config import *


class GeneEmbedder(nn.Module):

    def __init__(self, num_genes, embedding_dim=128):
        super(GeneEmbedder, self).__init__()
        
        self.embedding_mat = nn.Parameter(
            torch.randn(num_genes, 4, embedding_dim) * np.sqrt(2) / embedding_dim
        )

    def forward(self, gene_seq):
        """
        Arguments:
        ----------
        - gene_seq: (N, num_genes)

        Returns:
        --------
        - embedding_seq: (N, embedding_dim)
        """

        N, M = gene_seq.size()

        gene_seq = F.one_hot(gene_seq, 4).view(N, M, 4).float()

        embedding_mat = F.normalize(self.embedding_mat, dim=-1)
        embedding_mat = embedding_mat[None, :, :, :].expand(N, -1, -1, -1)
        embedding_seq = torch.einsum("nmi,nmid->nmd", gene_seq, embedding_mat)
        return embedding_seq

