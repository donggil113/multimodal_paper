r"""Raw chest-radiograph path: JPG loading and a compact CNN encoder.

As with the waveform module, the headline analysis uses CheXpert labels because
they are interpretable, universally available with MIMIC-CXR-JPG, and do not
require half a terabyte of pixels.  This module is the alternative for sites that
have the images, and it is also where a pretrained thoracic foundation model
would be plugged in -- :func:`embeddings_from_npz` accepts any externally
computed per-study embedding, so swapping in a stronger encoder changes one
argument and nothing else.

Substituting a stronger encoder *raises* :math:`I_{\mathcal V}` because it
enlarges the family.  That is the intended semantics, not a confound: usable
information is relative to the computation you can bring to bear, and reporting
which family produced a number is part of reporting the number.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from infogain.utils.logging import get_logger

log = get_logger("infogain.encoders.image")

DEFAULT_SIZE = 224


def load_cxr_jpg(path: str | Path, size: int = DEFAULT_SIZE) -> np.ndarray | None:
    """Load a MIMIC-CXR-JPG image as a ``(1, size, size)`` float array in [0, 1]."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("raw CXR needs `pip install Pillow`") from exc
    try:
        img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    except Exception as exc:  # pragma: no cover
        log.warning("unreadable CXR %s: %s", path, exc)
        return None
    return (np.asarray(img, dtype=np.float32) / 255.0)[None, :, :]


def normalise_cxr(x: np.ndarray) -> np.ndarray:
    """Per-image standardisation.

    Absolute intensity in a radiograph reflects exposure settings, not anatomy,
    so a model that keys on it learns the scanner rather than the patient.
    """
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=(-2, -1), keepdims=True)
    sd = x.std(axis=(-2, -1), keepdims=True) + 1e-6
    return (x - mu) / sd


class _ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, stride: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_out), nn.GELU(),
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out), nn.Dropout2d(dropout))
        self.skip = (nn.Identity() if (stride == 1 and ch_in == ch_out)
                     else nn.Sequential(nn.Conv2d(ch_in, ch_out, 1, stride=stride,
                                                  bias=False), nn.BatchNorm2d(ch_out)))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))


@dataclass
class CXREncoderConfig:
    channels: tuple[int, ...] = (24, 48, 96, 128, 160)
    emb_dim: int = 48
    dropout: float = 0.1


class CXREncoder(nn.Module):
    """Small residual CNN mapping ``(B, 1, 224, 224)`` to a fixed embedding."""

    def __init__(self, cfg: CXREncoderConfig | None = None):
        super().__init__()
        cfg = cfg or CXREncoderConfig()
        self.cfg = cfg
        layers: list[nn.Module] = [
            nn.Conv2d(1, cfg.channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(cfg.channels[0]), nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1)]
        ch = cfg.channels[0]
        for out_ch in cfg.channels:
            layers.append(_ConvBlock(ch, out_ch, 2 if out_ch != ch else 1, cfg.dropout))
            ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(ch, cfg.emb_dim), nn.LayerNorm(cfg.emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


@torch.no_grad()
def embed_images(encoder: CXREncoder, images: np.ndarray, batch_size: int = 32,
                 device: str = "cpu") -> np.ndarray:
    encoder = encoder.to(device).eval()
    out = []
    for s in range(0, len(images), batch_size):
        xb = torch.from_numpy(images[s:s + batch_size]).float().to(device)
        out.append(encoder(xb).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, encoder.cfg.emb_dim))


def embeddings_from_npz(path: str | Path) -> dict[int, np.ndarray]:
    """Load externally computed per-study embeddings.

    Expects an ``.npz`` with ``study_id`` ``(n,)`` and ``embedding`` ``(n, d)``.
    This is the hook for a pretrained chest-radiograph model: compute the
    embeddings once with whatever encoder you trust, save them here, and pass the
    result to :func:`infogain.data.mimic_modalities.extract_cxr`.
    """
    z = np.load(path)
    return {int(s): np.asarray(e, dtype=np.float32)
            for s, e in zip(z["study_id"], z["embedding"])}
