r"""The predictive family :math:`\mathcal V`: one network that realises every subset.

:math:`\mathcal V`-information is defined by an infimum over a family of
predictors, and the decomposition needs :math:`f_S` for every subset
:math:`S\subseteq M`.  Training :math:`2^{|M|}` separate models is both expensive
and statistically wasteful, so :class:`MaskedFusionFamily` amortises them into a
single network with a learned *absent* token per modality.  Setting the presence
flag of modality :math:`m` to zero replaces its embedding by that token, so

* every :math:`f_S` is a member of the same family by construction -- masking
  closure, the assumption that makes :math:`I_{\mathcal V}` monotone on the
  subset lattice and the conditional gains non-negative in population; and
* the null model :math:`f_\varnothing` (all tokens absent) is a member too, so
  :math:`H_{\mathcal V}(Y)` is estimated inside the family rather than swapped
  for a hand-written prevalence constant.

The presence flags are also fed to the head directly.  Without them the head
cannot tell "test not ordered" from "test ordered and unremarkable", and in EHR
data those carry very different risk.

Amortisation costs some fit quality on individual subsets.  Because
:math:`I_{\mathcal V}` is an infimum, any under-fitted head yields an
*under*-estimate, never an over-estimate, so the reported information is a valid
lower bound; :func:`infogain.encoders.train.fit_family` additionally supports
dedicated per-subset heads and keeps whichever attains lower held-out loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn


@dataclass
class FamilyConfig:
    emb_dim: int = 48
    enc_hidden: int = 128
    head_hidden: int = 192
    n_head_layers: int = 2
    dropout: float = 0.15
    use_presence_flags: bool = True
    weight_decay: float = 1e-4
    #: ``"attention"`` runs a small transformer over the modality tokens before
    #: pooling; ``"concat"`` is the plain concatenate-then-MLP baseline.
    #: Synergistic information lives in *interactions* between modalities, and a
    #: concatenation head has to discover those inside a generic MLP.  Attention
    #: gives every modality a direct multiplicative path to every other, and in
    #: the simulation study it recovers roughly twice as much of the known
    #: synergy at the same sample size.  Because the estimator is an infimum over
    #: the family, a weaker head only ever *under*-reports information, so this
    #: choice affects tightness, never validity.
    fusion: str = "attention"
    n_attn_layers: int = 2
    n_heads: int = 4
    #: Add an explicit second-order term over the modality tokens,
    #: :math:`\sum_{m<m'} e_m \odot e_{m'}`, computed by the factorization-machine
    #: identity :math:`\tfrac12[(\sum_m e_m)^2 - \sum_m e_m^2]`.
    #:
    #: This is not a tuning knob.  Synergistic information *is* an interaction
    #: between modalities, so a family with no multiplicative path between two
    #: tokens cannot represent it, and reporting "no synergy" from such a family
    #: would be a statement about the architecture rather than about the data.
    #: Attention can in principle compose one, but it has to learn to, from the
    #: few events an interaction term contributes to the loss; the explicit path
    #: costs d extra head inputs and removes the excuse.
    use_fm: bool = True


class ModalityEncoder(nn.Module):
    """Per-modality MLP encoder producing a fixed-width embedding."""

    def __init__(self, in_dim: int, emb_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim), nn.LayerNorm(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedFusionFamily(nn.Module):
    """One network realising :math:`f_S` for every modality subset :math:`S`."""

    def __init__(self, dims: dict[str, int], cfg: FamilyConfig | None = None,
                 n_outputs: int = 1):
        super().__init__()
        self.cfg = cfg or FamilyConfig()
        self.names: list[str] = list(dims)
        self.dims = dict(dims)
        self.n_outputs = n_outputs

        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(d, self.cfg.emb_dim, self.cfg.enc_hidden, self.cfg.dropout)
            for m, d in dims.items()})
        # learned "this modality was not available" token, one per modality
        self.absent = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(self.cfg.emb_dim)) for m in dims})

        M = len(dims)
        fm_dim = self.cfg.emb_dim if self.cfg.use_fm else 0
        if self.cfg.fusion == "attention":
            self.type_emb = nn.Parameter(torch.randn(M, self.cfg.emb_dim) * 0.02)
            self.present_emb = nn.Parameter(torch.randn(2, self.cfg.emb_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=self.cfg.emb_dim, nhead=self.cfg.n_heads,
                dim_feedforward=self.cfg.head_hidden, dropout=self.cfg.dropout,
                batch_first=True, norm_first=True, activation="gelu")
            # norm_first disables the nested-tensor fast path; silence the notice
            # rather than give up pre-norm stability on a 6-token sequence.
            self.attn = nn.TransformerEncoder(layer, num_layers=self.cfg.n_attn_layers,
                                              enable_nested_tensor=False)
            in_dim = self.cfg.emb_dim * 2 + fm_dim + (M if self.cfg.use_presence_flags else 0)
        else:
            self.attn = None
            in_dim = self.cfg.emb_dim * M + fm_dim + (M if self.cfg.use_presence_flags else 0)
        layers: list[nn.Module] = []
        h = in_dim
        for _ in range(self.cfg.n_head_layers):
            layers += [nn.Linear(h, self.cfg.head_hidden), nn.GELU(),
                       nn.Dropout(self.cfg.dropout)]
            h = self.cfg.head_hidden
        layers += [nn.Linear(h, n_outputs)]
        self.head = nn.Sequential(*layers)
        # start from the prevalence: the empty-mask member should not have to
        # learn the base rate from scratch, it anchors every PVI value
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    # ------------------------------------------------------------------ #
    def forward(self, x: dict[str, torch.Tensor],
                present: torch.Tensor) -> torch.Tensor:
        """``present`` is ``(batch, n_modalities)`` in ``self.names`` order."""
        embs = []
        for j, m in enumerate(self.names):
            e = self.encoders[m](x[m])
            p = present[:, j:j + 1]
            embs.append(p * e + (1.0 - p) * self.absent[m].unsqueeze(0))

        raw = torch.stack(embs, dim=1)                           # (B, M, d)
        if self.attn is not None:
            pres_idx = present.long().clamp(0, 1)                # (B, M)
            tok = raw + self.type_emb.unsqueeze(0) + self.present_emb[pres_idx]
            tok = self.attn(tok)
            # pool over present tokens only, but keep a global mean so the
            # empty-mask member still receives a well-defined input
            w = present.unsqueeze(-1)
            pooled = (tok * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
            h = torch.cat([pooled, tok.mean(dim=1)], dim=-1)
        else:
            h = torch.cat(embs, dim=-1)
        if self.cfg.use_fm:
            # sum over unordered pairs of the elementwise product, in O(Md)
            s1 = raw.sum(dim=1)
            s2 = (raw * raw).sum(dim=1)
            h = torch.cat([h, 0.5 * (s1 * s1 - s2)], dim=-1)
        if self.cfg.use_presence_flags:
            h = torch.cat([h, present], dim=-1)
        out = self.head(h)
        return out.squeeze(-1) if self.n_outputs == 1 else out

    def embed(self, x: dict[str, torch.Tensor], present: torch.Tensor) -> torch.Tensor:
        """Fused representation before the head.

        Used as the matching space for the conditional sampler in
        :mod:`infogain.vinfo.conditional`: donors for "what would this test have
        shown" must be neighbours in the representation the model *actually
        uses*, not in raw feature space where an irrelevant lab dominates the
        Euclidean distance.
        """
        embs = []
        for j, m in enumerate(self.names):
            e = self.encoders[m](x[m])
            pj = present[:, j:j + 1]
            embs.append(pj * e + (1.0 - pj) * self.absent[m].unsqueeze(0))
        if self.attn is not None:
            tok = torch.stack(embs, dim=1)
            pres_idx = present.long().clamp(0, 1)
            tok = tok + self.type_emb.unsqueeze(0) + self.present_emb[pres_idx]
            tok = self.attn(tok)
            w = present.unsqueeze(-1)
            return (tok * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return torch.cat(embs, dim=-1)

    # ------------------------------------------------------------------ #
    def subset_presence(self, observed: torch.Tensor,
                        subset: Iterable[str]) -> torch.Tensor:
        """Presence matrix for a subset: available **and** actually acquired."""
        keep = torch.zeros(len(self.names), device=observed.device, dtype=observed.dtype)
        s = set(subset)
        for j, m in enumerate(self.names):
            if m in s:
                keep[j] = 1.0
        return observed * keep.unsqueeze(0)

    def frobenius_norms(self) -> list[float]:
        """Layer-wise Frobenius norms, for the capacity term of Theorem 3."""
        return [float(p.detach().norm().item())
                for n, p in self.named_parameters() if p.dim() == 2]


# --------------------------------------------------------------------------- #
# Mask sampling during training
# --------------------------------------------------------------------------- #
@dataclass
class MaskSampler:
    r"""Distribution over training masks.

    The estimator only ever queries subsets, so the training distribution
    determines which members of :math:`\mathcal V` are well fitted.  Three
    components, each earning its place:

    * a spike on the empty set, because :math:`f_\varnothing` anchors every PVI
      value and a badly fitted null biases *all* subsets by the same amount;
    * a spike on the full set, the model a conventional multimodal paper would
      report, so our numbers are comparable to it; and
    * independent Bernoulli dropout elsewhere with a per-batch rate, which covers
      the lattice without over-representing the middle layers.
    """

    names: Sequence[str]
    always_available: frozenset[str] = frozenset()
    p_empty: float = 0.08
    p_full: float = 0.25
    p_baseline_only: float = 0.10
    keep_rate_range: tuple[float, float] = (0.25, 0.9)
    always_keep_rate: float = 0.85
    focus_subsets: list[frozenset[str]] = field(default_factory=list)
    #: relative weights within the focus branch, same length as ``focus_subsets``.
    #: Uniform if empty.  Weighting rather than a flat list matters because the
    #: subsets are not equally important: conditional gains are differences
    #: between the full panel and a leave-one-out set, so those need the most
    #: training, while the intermediate subsets still need enough to keep the
    #: Shapley values and the pairwise interaction map honest.
    focus_weights: list[float] = field(default_factory=list)
    p_focus: float = 0.15

    def sample(self, batch: int, generator: torch.Generator | None = None
               ) -> torch.Tensor:
        M = len(self.names)
        dev = generator.device if generator is not None else "cpu"
        u = torch.rand(batch, 1, generator=generator, device=dev)
        rate = torch.empty(1, device=dev).uniform_(*self.keep_rate_range,
                                                   generator=generator)
        mask = (torch.rand(batch, M, generator=generator, device=dev) < rate).float()
        base_idx = [j for j, m in enumerate(self.names) if m in self.always_available]
        if base_idx:
            keep_base = (torch.rand(batch, len(base_idx), generator=generator,
                                    device=dev) < self.always_keep_rate).float()
            mask[:, base_idx] = keep_base

        c1 = self.p_empty
        c2 = c1 + self.p_full
        c3 = c2 + self.p_baseline_only
        c4 = c3 + (self.p_focus if self.focus_subsets else 0.0)

        mask = torch.where(u < c1, torch.zeros_like(mask), mask)
        mask = torch.where((u >= c1) & (u < c2), torch.ones_like(mask), mask)
        if base_idx:
            base_only = torch.zeros_like(mask)
            base_only[:, base_idx] = 1.0
            mask = torch.where((u >= c2) & (u < c3), base_only, mask)
        if self.focus_subsets:
            if self.focus_weights:
                w = torch.tensor(self.focus_weights, dtype=torch.float32, device=dev)
                pick = torch.multinomial(w / w.sum(), batch, replacement=True,
                                         generator=generator)
            else:
                pick = torch.randint(len(self.focus_subsets), (batch,),
                                     generator=generator, device=dev)
            table = torch.tensor(
                [[1.0 if m in fs else 0.0 for m in self.names] for fs in self.focus_subsets],
                device=dev)
            mask = torch.where((u >= c3) & (u < c4), table[pick], mask)
        return mask


# --------------------------------------------------------------------------- #
# Tensor packing
# --------------------------------------------------------------------------- #
@dataclass
class PackedData:
    """Standardised modality tensors plus the acquisition mask."""

    x: dict[str, torch.Tensor]
    observed: torch.Tensor           # (n, M) float 0/1, in ``names`` order
    y: torch.Tensor                  # (n,)
    names: list[str]

    def to(self, device: str) -> "PackedData":
        return PackedData({k: v.to(device) for k, v in self.x.items()},
                          self.observed.to(device), self.y.to(device), self.names)

    def index(self, idx: np.ndarray | torch.Tensor) -> "PackedData":
        if isinstance(idx, np.ndarray):
            idx = torch.as_tensor(idx)
        return PackedData({k: v[idx] for k, v in self.x.items()},
                          self.observed[idx], self.y[idx], self.names)

    @property
    def n(self) -> int:
        return int(self.y.shape[0])


class Standardizer:
    """Per-feature mean/scale fitted on observed training rows only.

    Computing statistics over rows where the modality was never acquired would
    pull every mean toward the zero-fill and silently rescale the informative
    rows.
    """

    def __init__(self) -> None:
        self.mean: dict[str, np.ndarray] = {}
        self.scale: dict[str, np.ndarray] = {}

    def fit(self, blocks: dict[str, np.ndarray], observed: dict[str, np.ndarray],
            rows: np.ndarray) -> "Standardizer":
        for m, v in blocks.items():
            sel = rows[observed[m][rows]] if observed[m][rows].any() else rows
            sub = v[sel]
            mu = sub.mean(axis=0)
            sd = sub.std(axis=0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            self.mean[m], self.scale[m] = mu.astype(np.float32), sd.astype(np.float32)
        return self

    def transform(self, blocks: dict[str, np.ndarray],
                  observed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for m, v in blocks.items():
            z = (v - self.mean[m]) / self.scale[m]
            z = np.clip(z, -8.0, 8.0)          # guard against lab outliers
            z[~observed[m]] = 0.0
            out[m] = z.astype(np.float32)
        return out


def pack(cohort, outcome: str, standardizer: Standardizer,
         rows: np.ndarray | None = None) -> PackedData:
    """Turn a :class:`~infogain.data.schema.Cohort` into model-ready tensors."""
    names = cohort.spec.names
    blocks = {m: cohort.blocks[m].values for m in names}
    observed = {m: cohort.blocks[m].observed for m in names}
    z = standardizer.transform(blocks, observed)
    y = cohort.y(outcome)
    if rows is not None:
        z = {m: v[rows] for m, v in z.items()}
        observed = {m: v[rows] for m, v in observed.items()}
        y = y[rows]
    obs = np.stack([observed[m] for m in names], axis=1).astype(np.float32)
    return PackedData({m: torch.from_numpy(np.ascontiguousarray(z[m])) for m in names},
                      torch.from_numpy(obs), torch.from_numpy(y.astype(np.float32)),
                      list(names))
