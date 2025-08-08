import argparse
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


# MLP class for Self-Supervised Learning Projector & Predictor
class MLPHead(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super(MLPHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(in_features, out_features, bias=True))

    def forward(self, x: torch.Tensor):
        return self.mlp(x)
