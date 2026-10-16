"""MLP and focal loss used by the current frozen-Clay classifier."""
import torch
from torch import nn
from torch.nn import functional as F

EMBED_DIM = 2048

class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def sigmoid_focal_loss(logits, targets, alpha=0.5, gamma=2.0):
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t).pow(gamma) * ce).mean()
