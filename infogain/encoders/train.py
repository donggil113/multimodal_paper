r"""Cross-fitted training of the masked family and out-of-fold PVI extraction.

Two statistical requirements shape this module.

**Held-out evaluation, for everyone.**  :math:`\hat I_{\mathcal V}` is only
meaningful on data the predictor did not see, but per-patient gains
:math:`\Delta_i` are wanted for *every* patient, not just a test split.
:math:`K`-fold cross-fitting resolves the tension: each patient is scored by the
models that never saw them, and the union of the folds covers the cohort.  Folds
are grouped by ``subject_id``.

**Estimator variance is not sampling variance.**  Re-running with a different
seed moves :math:`\hat I_{\mathcal V}` by an amount that has nothing to do with
patient sampling.  Every fold is therefore trained under several seeds, and the
between-seed spread is carried through to the confidence intervals in
:mod:`infogain.vinfo.core` instead of being averaged away silently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from infogain.encoders.fusion import (
    FamilyConfig, MaskedFusionFamily, MaskSampler, PackedData, Standardizer, pack,
)
from infogain.utils.logging import get_logger
from infogain.utils.seed import set_seed
from infogain.vinfo.core import (
    DEFAULT_PVI_CLIP, label_logprob, pointwise_v_information, subset_key,
)

log = get_logger("infogain.encoders.train")


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 512
    lr: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 12
    n_folds: int = 5
    seeds: tuple[int, ...] = (0, 1, 2)
    inner_val_frac: float = 0.15
    device: str = "cpu"
    family: FamilyConfig = field(default_factory=FamilyConfig)
    grad_clip: float = 5.0
    min_epochs: int = 15
    verbose: bool = False
    #: How :math:`H_{\mathcal V}(Y)` -- the PVI anchor -- is obtained.
    #: ``"prevalence"`` uses the training-fold base rate, which is the exact
    #: minimiser over the constant sub-family and therefore attains the
    #: infimum that defines :math:`H_{\mathcal V}(Y)`.  ``"empty_mask"`` uses the
    #: network's own all-absent member, which is *also* in the family but is
    #: only approximately optimal; any slack in it inflates every level estimate
    #: by the same amount (it cancels in differences, but the paper reports
    #: levels too).  Default to the provable choice and report the gap.
    null_mode: str = "prevalence"


# --------------------------------------------------------------------------- #
# Subset families
# --------------------------------------------------------------------------- #
def default_subsets(names: Sequence[str], baseline: Iterable[str]
                    ) -> list[frozenset[str]]:
    """The lattice the decomposition actually queries.

    Everything is measured relative to the free baseline :math:`S_0`, so we need
    :math:`S_0\\cup T` for every :math:`T` in the powerset of the orderable
    modalities, plus the empty set (anchors PVI) and each orderable modality
    *alone* (its stand-alone value, the number a single-modality paper reports).
    """
    from itertools import combinations

    base = frozenset(baseline)
    orderable = [n for n in names if n not in base]
    out: list[frozenset[str]] = [frozenset()]
    for r in range(len(orderable) + 1):
        for c in combinations(sorted(orderable), r):
            out.append(base | frozenset(c))
    for m in orderable:
        out.append(frozenset({m}))
    seen, uniq = set(), []
    for s in out:
        k = subset_key(s)
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


# --------------------------------------------------------------------------- #
# Single-model training
# --------------------------------------------------------------------------- #
def _train_one(data: PackedData, dims: dict[str, int], sampler: MaskSampler,
               cfg: TrainConfig, train_idx: np.ndarray, val_idx: np.ndarray,
               seed: int) -> tuple[MaskedFusionFamily, float]:
    set_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = MaskedFusionFamily(dims, cfg.family).to(cfg.device)

    # anchor the empty-mask member at the training prevalence
    prev = float(data.y[torch.as_tensor(train_idx)].mean().clamp(1e-4, 1 - 1e-4))
    with torch.no_grad():
        model.head[-1].bias.fill_(math.log(prev / (1 - prev)))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    lossf = nn.BCEWithLogitsLoss()
    gen = torch.Generator(device=cfg.device).manual_seed(seed)

    tr = data.index(train_idx)
    va = data.index(val_idx)
    # fixed validation masks so early stopping is not chasing mask noise
    va_mask = sampler.sample(va.n, torch.Generator(device=cfg.device).manual_seed(12345))
    va_present = va.observed * va_mask

    best_state, best_loss, bad = None, float("inf"), 0
    n = tr.n
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen, device=cfg.device)
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            xb = {m: v[idx] for m, v in tr.x.items()}
            ob = tr.observed[idx]
            mask = sampler.sample(idx.numel(), gen)
            logits = model(xb, ob * mask)
            loss = lossf(logits, tr.y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(va.x, va_present), va.y))
        if vl < best_loss - 1e-5:
            best_loss, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience and epoch >= cfg.min_epochs:
                break
        if cfg.verbose and epoch % 10 == 0:
            log.info("  seed %d epoch %3d val %.5f", seed, epoch, vl)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


# --------------------------------------------------------------------------- #
# Cross-fitted family
# --------------------------------------------------------------------------- #
@dataclass
class CrossFittedFamily:
    """Out-of-fold probabilities for every requested subset, per seed."""

    outcome: str
    names: list[str]
    baseline: frozenset[str]
    subsets: list[frozenset[str]]
    probs: dict[str, np.ndarray]      # subset key -> (n_seeds, n) OOF probabilities
    p_null: np.ndarray                # (n,) fold-train prevalence: the PVI anchor
    y: np.ndarray
    fold_id: np.ndarray
    val_losses: dict[int, list[float]]
    frob_norms: list[float]
    input_radius: float
    n_seeds: int
    null_mode: str = "prevalence"
    #: ``fold_models[k][seed_index]`` -- retained only when ``keep_models=True``.
    #: Needed by the per-patient estimator, which must score each patient with a
    #: model that never saw them.
    fold_models: list[list] = field(default_factory=list)
    standardizers: list = field(default_factory=list)
    rows: np.ndarray | None = None
    dims: dict[str, int] = field(default_factory=dict)

    # -- probabilities ------------------------------------------------------ #
    def p(self, subset: Iterable[str], seed_index: int | None = None) -> np.ndarray:
        arr = self.probs[subset_key(subset)]
        return arr[seed_index] if seed_index is not None else arr.mean(axis=0)

    def has(self, subset: Iterable[str]) -> bool:
        return subset_key(subset) in self.probs

    # -- pointwise V-information -------------------------------------------- #
    def null_probs(self, seed_index: int | None = None) -> np.ndarray:
        return (self.p_null if self.null_mode == "prevalence"
                else self.p(frozenset(), seed_index))

    def pvi(self, subset: Iterable[str], seed_index: int | None = None,
            clip: float = DEFAULT_PVI_CLIP, rows: np.ndarray | None = None
            ) -> np.ndarray:
        r"""Per-patient :math:`\mathrm{pvi}_i(S)` in bits, against the null member."""
        y = self.y if rows is None else self.y[rows]
        pm = self.p(subset, seed_index)
        p0 = self.null_probs(seed_index)
        if rows is not None:
            pm, p0 = pm[rows], p0[rows]
        return pointwise_v_information(label_logprob(pm, y), label_logprob(p0, y),
                                       clip=clip)

    def null_gap_bits(self) -> float:
        """Excess held-out cross-entropy of the empty-mask member over the constant.

        A large positive value means the network's all-absent member has not
        converged to the base rate; it is reported with every run because it is
        the amount by which an ``empty_mask`` null would inflate every level.
        """
        from infogain.vinfo.core import LOG2

        ce_mask = -label_logprob(self.p(frozenset()), self.y).mean() / LOG2
        ce_const = -label_logprob(self.p_null, self.y).mean() / LOG2
        return float(ce_mask - ce_const)

    def pvi_all_seeds(self, subset: Iterable[str], clip: float = DEFAULT_PVI_CLIP,
                      rows: np.ndarray | None = None) -> np.ndarray:
        return np.stack([self.pvi(subset, s, clip, rows) for s in range(self.n_seeds)])

    def seed_means(self, subset: Iterable[str], rows: np.ndarray | None = None
                   ) -> np.ndarray:
        return self.pvi_all_seeds(subset, rows=rows).mean(axis=1)


def fit_family(cohort, outcome: str, cfg: TrainConfig | None = None,
               subsets: Sequence[frozenset[str]] | None = None,
               rows: np.ndarray | None = None,
               keep_models: bool = False,
               y_override: np.ndarray | None = None) -> CrossFittedFamily:
    """Cross-fit the masked family and return out-of-fold subset probabilities."""
    from sklearn.model_selection import GroupKFold

    cfg = cfg or TrainConfig()
    names = cohort.spec.names
    baseline = cohort.spec.baseline
    subsets = list(subsets) if subsets is not None else default_subsets(names, baseline)

    evaluable = cohort.evaluable(outcome)
    rows = np.where(evaluable)[0] if rows is None else np.asarray(rows)
    rows = rows[evaluable[rows]]
    n = len(rows)
    groups = cohort.index["subject_id"].to_numpy()[rows]
    y = (cohort.y(outcome) if y_override is None else np.asarray(y_override))[rows]
    log.info("fit_family(%s): n=%d, prevalence=%.3f%%, %d subsets, %d folds x %d seeds",
             outcome, n, 100 * y.mean(), len(subsets), cfg.n_folds, len(cfg.seeds))

    dims = {m: cohort.blocks[m].dim for m in names}
    sampler = MaskSampler(names=names, always_available=frozenset(baseline),
                          focus_subsets=[s for s in subsets if s])

    probs = {subset_key(s): np.zeros((len(cfg.seeds), n), dtype=np.float64)
             for s in subsets}
    p_null = np.zeros(n, dtype=np.float64)
    fold_id = np.zeros(n, dtype=int)
    val_losses: dict[int, list[float]] = {s: [] for s in cfg.seeds}
    frob: list[float] = []
    radius = 0.0
    fold_models: list[list] = []
    standardizers: list = []

    gkf = GroupKFold(n_splits=cfg.n_folds)
    for k, (tr_pos, te_pos) in enumerate(gkf.split(np.zeros(n), y, groups)):
        fold_id[te_pos] = k
        p_null[te_pos] = float(np.clip(y[tr_pos].mean(), 1e-5, 1 - 1e-5))
        std = Standardizer().fit(
            {m: cohort.blocks[m].values for m in names},
            {m: cohort.blocks[m].observed for m in names},
            rows[tr_pos])
        data = pack(cohort, outcome, std, rows).to(cfg.device)
        if y_override is not None:
            data = PackedData(data.x, data.observed,
                              torch.from_numpy(y.astype(np.float32)), data.names)
        radius = max(radius, float(torch.stack(
            [v.norm(dim=1).max() for v in data.x.values()]).max()))

        # inner validation split, grouped, for early stopping
        g_tr = groups[tr_pos]
        uniq = np.unique(g_tr)
        rng = np.random.default_rng(1000 + k)
        rng.shuffle(uniq)
        n_val = max(1, int(cfg.inner_val_frac * uniq.size))
        val_groups = set(uniq[:n_val].tolist())
        inner_val = tr_pos[np.isin(g_tr, list(val_groups))]
        inner_tr = tr_pos[~np.isin(g_tr, list(val_groups))]

        seed_models = []
        for si, seed in enumerate(cfg.seeds):
            model, vl = _train_one(data, dims, sampler, cfg, inner_tr, inner_val,
                                   seed=seed + 100 * k)
            val_losses[seed].append(vl)
            if keep_models:
                seed_models.append(model)
            if k == 0 and si == 0:
                frob = model.frobenius_norms()
            te = data.index(te_pos)
            with torch.no_grad():
                for s in subsets:
                    pres = model.subset_presence(te.observed, s)
                    probs[subset_key(s)][si, te_pos] = torch.sigmoid(
                        model(te.x, pres)).cpu().numpy()
        if keep_models:
            fold_models.append(seed_models)
            standardizers.append(std)
        log.info("  fold %d/%d done (val loss %.5f)", k + 1, cfg.n_folds, vl)

    cf = CrossFittedFamily(
        outcome=outcome, names=list(names), baseline=frozenset(baseline),
        subsets=subsets, probs=probs, p_null=p_null, y=y, fold_id=fold_id,
        val_losses=val_losses, frob_norms=frob, input_radius=radius,
        n_seeds=len(cfg.seeds), null_mode=cfg.null_mode,
        fold_models=fold_models, standardizers=standardizers, rows=rows, dims=dims)
    log.info("null-model gap (empty-mask minus constant): %+.5f bits", cf.null_gap_bits())
    return cf
