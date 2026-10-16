"""Load reusable frozen-Clay classifier heads and apply saved thresholds.

Callers must supply embeddings made with the checkpoint's encoder protocol.
"""
from pathlib import Path
import torch
from torch import nn
try:
    from .model_head import MLPHead
    from .encoder_geometry import validate_geometry
except ImportError:
    from model_head import MLPHead
    from encoder_geometry import validate_geometry


def make_head(architecture):
    if architecture=='mlp':return MLPHead()
    if architecture=='small_mlp':return nn.Sequential(nn.Linear(2048,32),nn.ReLU(),nn.Dropout(.3),nn.Linear(32,1))
    raise ValueError(f'Unknown architecture: {architecture}')


def load_head(path,device='cpu'):
    checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
    if checkpoint.get('format_version')!=1:raise ValueError('Unsupported final checkpoint format')
    validate_geometry(checkpoint.get('encoder', {}), checkpoint=True)
    model=make_head(checkpoint['architecture']);model.load_state_dict(checkpoint['state_dict']);model.to(device).eval()
    return model,checkpoint


@torch.inference_mode()
def predict_embeddings(model,checkpoint,embeddings):
    device=next(model.parameters()).device
    x=torch.as_tensor(embeddings,dtype=torch.float32,device=device)
    if x.ndim!=2 or x.shape[1]!=2048:raise ValueError('Expected N x 2048 CLS-plus-pooled-patch embeddings')
    probabilities=torch.sigmoid(model(x).reshape(-1))
    return probabilities,probabilities.ge(checkpoint['threshold'])
