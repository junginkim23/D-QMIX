import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class BYOLLoss(nn.Module):
    def __init__(self):
        super(BYOLLoss, self).__init__()

    @staticmethod
    def calculate_loss(pred, true):
        pred = F.normalize(pred, dim=-1)
        true = F.normalize(true, dim=-1)

        return 2 - 2 * (pred * true).sum(dim=-1)

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    @staticmethod
    def calculate_loss(pred,true):
        f_x1 = F.normalize(pred.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(true.float(), p=2., dim=-1, eps=1e-3)
        # Gradients of norrmalized L2 loss and cosine similiarity are proportional.
        # See: https://stats.stackexchange.com/a/146279
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)

        return loss
