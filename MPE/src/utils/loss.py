import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class MSFDMLoss(nn.Module):
    def __init__(self):
        super(MSFDMLoss, self).__init__()

    @staticmethod
    def calculate_loss(pred, true):
        pred = F.normalize(pred, dim=-1)
        true = F.normalize(true, dim=-1)

        return ((pred - true) ** 2).mean(dim=-1)
