import math, torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from torch import nn


def load_dino(device="cuda"):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    return model


@torch.no_grad()
def dino_per_pixel_features(img_bchw, model, l2_norm=True, patch=14, snap="floor"):
    """
    img_bchw: [B,3,H,W] in [0,1] or [0,255]
    returns:  [B,C,H,W] per-pixel features aligned to the ORIGINAL H×W
    """
    dev = next(model.parameters()).device
    x = img_bchw.to(dev, dtype=torch.float32)
    if x.max() > 1.5: x = x / 255.0
    x = normalize(x, [0.485,0.456,0.406], [0.229,0.224,0.225])

    H, W = x.shape[-2:]
    def snap_dim(n):
        if snap == "floor": return (n // patch) * patch
        if snap == "ceil":  return math.ceil(n / patch) * patch
        return round(n / patch) * patch  # "round"

    H14, W14 = snap_dim(H), snap_dim(W)
    if (H14, W14) != (H, W):
        x_in = F.interpolate(x, size=(H14, W14), mode="bicubic", align_corners=False)
    else:
        x_in = x

    out = model.forward_features(x_in)
    toks = out["x_norm_patchtokens"]              # [B, N, C]
    B, N, C = toks.shape
    Sh, Sw = H14 // patch, W14 // patch
    fmap = toks.view(B, Sh, Sw, C).permute(0, 3, 1, 2).contiguous()  # [B,C,Sh,Sw]
    if l2_norm:
        fmap = F.normalize(fmap, dim=1)

    # return per-pixel features at original resolution
    perpix = F.interpolate(fmap, size=(H, W), mode="bilinear", align_corners=False)
    return perpix


class TinyDecoder(nn.Module):
    def __init__(self, in_dim=30, out_dim=384, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):  # x: [..., in_dim]
        return self.net(x)