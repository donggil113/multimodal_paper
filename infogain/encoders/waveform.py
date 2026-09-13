r"""Raw 12-lead ECG path: waveform loading and a 1-D residual encoder.

The headline analysis uses MIMIC-IV-ECG's machine measurements, which are
interpretable and need no waveform download.  This module is the alternative
path for sites that have the ~90 GB of signal data and want the extra
information a learned representation extracts -- and it exists precisely so the
paper can report how much extra that is, rather than assert it.

The encoder is a compact 1-D ResNet over 10 s x 12 leads at 100 Hz.  It is
trained *self-supervised-free*: embeddings are produced by fitting the encoder
jointly with the outcome head inside the same masked family, so the
:math:`\mathcal V`-information semantics are unchanged -- the family is still
"everything this architecture can express given these inputs".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from infogain.utils.logging import get_logger

log = get_logger("infogain.encoders.waveform")

TARGET_FS = 100
TARGET_SECONDS = 10
N_LEADS = 12
LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_ecg_record(path: str | Path, target_fs: int = TARGET_FS,
                    seconds: int = TARGET_SECONDS) -> np.ndarray | None:
    """Read one WFDB record into ``(12, target_fs*seconds)`` millivolts.

    Returns ``None`` rather than raising when a record is unreadable: MIMIC-IV-ECG
    contains a small number of truncated files, and a modality that fails to load
    is simply "not acquired" for that patient, which the rest of the pipeline
    already handles.
    """
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("raw ECG needs `pip install wfdb`") from exc

    try:
        rec = wfdb.rdrecord(str(Path(path).with_suffix("")))
    except Exception as exc:  # pragma: no cover - corrupt records exist
        log.warning("unreadable ECG %s: %s", path, exc)
        return None

    sig = np.asarray(rec.p_signal, dtype=np.float32)          # (n_samples, n_leads)
    names = [str(s).strip() for s in (rec.sig_name or [])]
    out = np.zeros((N_LEADS, sig.shape[0]), dtype=np.float32)
    for i, lead in enumerate(LEAD_ORDER):
        if lead in names:
            out[i] = sig[:, names.index(lead)]
        elif lead.lower() in [n.lower() for n in names]:
            out[i] = sig[:, [n.lower() for n in names].index(lead.lower())]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    fs = int(getattr(rec, "fs", target_fs))
    if fs != target_fs:
        n_out = int(round(out.shape[1] * target_fs / fs))
        idx = np.linspace(0, out.shape[1] - 1, n_out)
        out = np.stack([np.interp(idx, np.arange(out.shape[1]), ch) for ch in out])
    want = target_fs * seconds
    if out.shape[1] >= want:
        start = (out.shape[1] - want) // 2
        out = out[:, start:start + want]
    else:
        out = np.pad(out, ((0, 0), (0, want - out.shape[1])))
    return out.astype(np.float32)


def normalise_ecg(x: np.ndarray, clip_mv: float = 5.0) -> np.ndarray:
    """Per-lead baseline removal and amplitude clipping.

    Baseline wander is a recording artefact, not physiology; leaving it in lets
    the encoder key on the recording session rather than the heart.
    """
    x = np.asarray(x, dtype=np.float32)
    x = x - np.median(x, axis=-1, keepdims=True)
    return np.clip(x, -clip_mv, clip_mv)


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class _ResBlock1d(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, stride: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(ch_in, ch_out, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(ch_out)
        self.conv2 = nn.Conv1d(ch_out, ch_out, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(ch_out)
        self.drop = nn.Dropout(dropout)
        self.skip = (nn.Identity() if (stride == 1 and ch_in == ch_out)
                     else nn.Sequential(nn.Conv1d(ch_in, ch_out, 1, stride=stride,
                                                  bias=False), nn.BatchNorm1d(ch_out)))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return self.act(h + self.skip(x))


@dataclass
class ECGEncoderConfig:
    channels: tuple[int, ...] = (32, 64, 96, 128)
    strides: tuple[int, ...] = (2, 2, 2, 2)
    emb_dim: int = 48
    dropout: float = 0.1


class ECGWaveformEncoder(nn.Module):
    """1-D residual CNN mapping ``(B, 12, 1000)`` to a fixed embedding.

    Drop-in replacement for the tabular :class:`ModalityEncoder`: same output
    width, same masking semantics, so the family stays closed under masking and
    every theorem continues to apply unchanged.
    """

    def __init__(self, cfg: ECGEncoderConfig | None = None, n_leads: int = N_LEADS):
        super().__init__()
        cfg = cfg or ECGEncoderConfig()
        self.cfg = cfg
        layers: list[nn.Module] = [
            nn.Conv1d(n_leads, cfg.channels[0], 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(cfg.channels[0]), nn.GELU()]
        ch = cfg.channels[0]
        for out_ch, stride in zip(cfg.channels, cfg.strides):
            layers.append(_ResBlock1d(ch, out_ch, stride, cfg.dropout))
            ch = out_ch
        self.backbone = nn.Sequential(*layers)
        # concatenating average and max pooling keeps both "how much of the trace
        # looks like this" and "did it ever happen" -- an isolated ectopic beat is
        # invisible to mean pooling alone
        self.proj = nn.Sequential(nn.Linear(2 * ch, cfg.emb_dim),
                                  nn.LayerNorm(cfg.emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.proj(pooled)


@torch.no_grad()
def embed_records(encoder: ECGWaveformEncoder, waveforms: np.ndarray,
                  batch_size: int = 64, device: str = "cpu") -> np.ndarray:
    """Batch-embed an ``(n, 12, T)`` array into ``(n, emb_dim)``."""
    encoder = encoder.to(device).eval()
    out = []
    for s in range(0, len(waveforms), batch_size):
        xb = torch.from_numpy(waveforms[s:s + batch_size]).float().to(device)
        out.append(encoder(xb).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, encoder.cfg.emb_dim))


def build_waveform_matrix(record_paths: dict[int, str], ecg_root: str | Path,
                          n_rows: int, row_to_study: dict[int, int],
                          seconds: int = TARGET_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    """Materialise ``(n_rows, 12, T)`` waveforms plus an observed mask.

    ``record_paths`` maps ``study_id`` to the path column of
    ``mimic-iv-ecg/record_list.csv``; ``row_to_study`` maps cohort rows to the
    study selected by :func:`infogain.data.mimic_modalities.extract_ecg`.
    """
    T = TARGET_FS * seconds
    X = np.zeros((n_rows, N_LEADS, T), dtype=np.float32)
    observed = np.zeros(n_rows, dtype=bool)
    for row, study in row_to_study.items():
        rel = record_paths.get(int(study))
        if rel is None:
            continue
        sig = load_ecg_record(Path(ecg_root) / rel, seconds=seconds)
        if sig is None:
            continue
        X[row] = normalise_ecg(sig)
        observed[row] = True
    log.info("loaded %d/%d ECG waveforms", int(observed.sum()), n_rows)
    return X, observed
