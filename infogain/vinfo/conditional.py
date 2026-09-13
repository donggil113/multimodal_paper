r"""Per-patient conditional gain :math:`\Delta_i(m\mid S)` -- the deployable score.

The cohort quantity :math:`U(m\mid S)` answers "is this test worth ordering *on
average*".  The clinical question is narrower and harder: **for this patient,
right now, before the test is done**, how much would it change what we believe?

.. math::
    \Delta_i(m\mid S) \;=\; \mathbb{E}_{x_m\sim q(\cdot\mid x_i^S)}\;
    \mathrm{KL}\!\big(f_{S\cup m}(\cdot\mid x_i^S, x_m)\,\big\|\,f_S(\cdot\mid x_i^S)\big)

This is the expected information gain of Bayesian experimental design, evaluated
inside :math:`\mathcal V` rather than under a generative model of the patient.
It is prospective by construction: it never touches the realised
:math:`x_m` or the label, so it is computable at the bedside.  Averaged over
patients it returns the cohort conditional gain -- an identity we check rather
than assert (:func:`consistency_check`).

**Where the counterfactual test result comes from.**  We need to integrate over
what the test *would have* shown.  Rather than fit a generative model of a
70-dimensional laboratory panel, we resample real results from patients who did
receive the test and who look alike in the fused representation the outcome
model actually uses (:func:`KNNConditionalSampler`).  This is a conditional
bootstrap: it inherits the true joint structure of the modality, needs no
density estimate, and degrades gracefully -- if no similar patient ever received
the test, the overlap diagnostic says so instead of extrapolating.

**What has to be true.**  Ordering is a clinical decision, so acquisition is not
random.  Identification of :math:`\Delta_i` for patients who did *not* get the
test needs
(i) **MAR** -- ordering independent of the result given the observed context, and
(ii) **overlap** -- similar patients who did get it.
Both are testable to a degree and both are reported:
:func:`overlap_diagnostics` quantifies (ii) directly, and the MNAR sensitivity
analysis in :mod:`infogain.clinical.policy` bounds the damage from violating (i).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from infogain.encoders.fusion import pack
from infogain.utils.logging import get_logger

log = get_logger("infogain.vinfo.conditional")
LN2 = float(np.log(2.0))


@dataclass
class GainConfig:
    n_samples: int = 24          # Monte-Carlo draws of the counterfactual result
    n_neighbors: int = 40        # donor pool size per patient
    sampler: str = "knn"         # {"knn", "gaussian", "marginal"}
    chunk: int = 4096
    seed: int = 0
    min_donors: int = 10


def _kl_bern(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


# --------------------------------------------------------------------------- #
# Samplers for the unacquired modality
# --------------------------------------------------------------------------- #
class KNNConditionalSampler:
    """Conditional bootstrap: donate real results from similar, tested patients."""

    def __init__(self, donor_emb: np.ndarray, donor_x: np.ndarray,
                 n_neighbors: int = 40, seed: int = 0):
        from sklearn.neighbors import NearestNeighbors

        self.donor_x = donor_x
        self.k = int(min(n_neighbors, len(donor_x)))
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(donor_emb)
        self.rng = np.random.default_rng(seed)

    def sample(self, query_emb: np.ndarray, n_samples: int) -> np.ndarray:
        """``(n_samples, n_query, d_m)`` counterfactual results."""
        _, idx = self.nn.kneighbors(query_emb)             # (n_query, k)
        pick = self.rng.integers(0, self.k, size=(n_samples, len(query_emb)))
        chosen = np.take_along_axis(idx[None, :, :], pick[:, :, None], axis=2)[:, :, 0]
        return self.donor_x[chosen]

    def neighbor_distances(self, query_emb: np.ndarray) -> np.ndarray:
        d, _ = self.nn.kneighbors(query_emb)
        return d.mean(axis=1)


class GaussianConditionalSampler:
    """Linear-Gaussian conditional in the modality's own feature space.

    A smooth parametric alternative to the conditional bootstrap: regress the
    modality's features on the fused context embedding, then sample around the
    prediction with the empirical residual covariance (diagonal).  Included
    mainly so the paper can show the per-patient scores are not an artefact of
    one sampling scheme.
    """

    def __init__(self, donor_emb: np.ndarray, donor_x: np.ndarray, seed: int = 0,
                 ridge: float = 1.0):
        from sklearn.linear_model import Ridge

        self.model = Ridge(alpha=ridge).fit(donor_emb, donor_x)
        resid = donor_x - self.model.predict(donor_emb)
        self.sd = resid.std(axis=0) + 1e-6
        self.rng = np.random.default_rng(seed)

    def sample(self, query_emb: np.ndarray, n_samples: int) -> np.ndarray:
        mu = self.model.predict(query_emb)
        noise = self.rng.normal(0.0, 1.0, size=(n_samples,) + mu.shape) * self.sd
        return mu[None, :, :] + noise


class MarginalSampler:
    """Ignore the context and resample the modality marginally.

    Deliberately mis-specified: it is the control that shows how much of
    :math:`\\Delta_i`'s patient-to-patient variation comes from *conditioning*
    rather than from the modality's overall variability.
    """

    def __init__(self, donor_x: np.ndarray, seed: int = 0):
        self.donor_x = donor_x
        self.rng = np.random.default_rng(seed)

    def sample(self, query_emb: np.ndarray, n_samples: int) -> np.ndarray:
        idx = self.rng.integers(0, len(self.donor_x),
                                size=(n_samples, len(query_emb)))
        return self.donor_x[idx]


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
@dataclass
class PatientGainResult:
    modality: str
    context: frozenset[str]
    delta_bits: np.ndarray            # (n,) prospective expected gain
    base_risk: np.ndarray             # (n,) P(Y=1 | x_S)
    donor_distance: np.ndarray        # (n,) mean distance to donors (overlap proxy)
    observed: np.ndarray              # (n,) was the modality actually acquired
    n_donors: int
    sampler: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "modality": self.modality, "delta_bits": self.delta_bits,
            "base_risk": self.base_risk, "donor_distance": self.donor_distance,
            "modality_observed": self.observed})

    def summary(self) -> dict:
        d = self.delta_bits
        return {"modality": self.modality, "n": int(d.size),
                "mean_bits": float(d.mean()), "median_bits": float(np.median(d)),
                "p90_bits": float(np.quantile(d, 0.9)),
                "p99_bits": float(np.quantile(d, 0.99)),
                "frac_above_0.01": float((d > 0.01).mean()),
                "gini": float(_gini(d)), "sampler": self.sampler}


def _gini(x: np.ndarray) -> float:
    """Concentration of the gain across patients: 0 = uniform, 1 = one patient.

    The number the paper leans on: if benefit were uniform, patient-level
    targeting could not beat a blanket policy, and the whole exercise would be
    pointless.
    """
    x = np.sort(np.clip(np.asarray(x, dtype=np.float64), 0, None))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def patient_expected_gain(cf, cohort, modality: str,
                          context: Iterable[str] | None = None,
                          cfg: GainConfig | None = None,
                          seed_index: int = 0) -> PatientGainResult:
    r"""Cross-fitted per-patient :math:`\Delta_i(m\mid S)` in bits.

    Each patient is scored by the fold model that never trained on them, and the
    donor pool for their counterfactual test result is drawn from that model's
    *training* folds only -- otherwise a patient could donate a result to
    themselves and the score would leak.
    """
    cfg = cfg or GainConfig()
    if not cf.fold_models:
        raise ValueError("fit_family(..., keep_models=True) is required")
    ctx = frozenset(context) if context is not None else (frozenset(cf.names) - {modality})
    ctx = ctx - {modality}

    rows = cf.rows
    n = len(rows)
    delta = np.zeros(n)
    base = np.zeros(n)
    dist = np.full(n, np.nan)
    mod_obs = cohort.blocks[modality].observed[rows]
    n_donor_total = 0

    for k, std in enumerate(cf.standardizers):
        model = cf.fold_models[k][min(seed_index, len(cf.fold_models[k]) - 1)]
        model.eval()
        te = np.where(cf.fold_id == k)[0]
        tr = np.where(cf.fold_id != k)[0]
        data = pack(cohort, cf.outcome, std, rows)

        with torch.no_grad():
            pres_ctx = model.subset_presence(data.observed, ctx)
            emb_all = model.embed(data.x, pres_ctx).numpy()

        donor_pos = tr[cohort.blocks[modality].observed[rows[tr]]]
        if donor_pos.size < cfg.min_donors:
            log.warning("fold %d: only %d donors for %s; skipping",
                        k, donor_pos.size, modality)
            continue
        n_donor_total += donor_pos.size
        donor_x = data.x[modality].numpy()[donor_pos]

        if cfg.sampler == "knn":
            sampler = KNNConditionalSampler(emb_all[donor_pos], donor_x,
                                            cfg.n_neighbors, cfg.seed + k)
            dist[te] = sampler.neighbor_distances(emb_all[te])
        elif cfg.sampler == "gaussian":
            sampler = GaussianConditionalSampler(emb_all[donor_pos], donor_x, cfg.seed + k)
        else:
            sampler = MarginalSampler(donor_x, cfg.seed + k)

        for start in range(0, te.size, cfg.chunk):
            sel = te[start:start + cfg.chunk]
            x_chunk = {m: data.x[m][sel] for m in data.names}
            obs_chunk = data.observed[sel].clone()
            with torch.no_grad():
                p_base = torch.sigmoid(
                    model(x_chunk, model.subset_presence(obs_chunk, ctx))).numpy()
            base[sel] = p_base

            draws = sampler.sample(emb_all[sel], cfg.n_samples)   # (S, b, d)
            obs_plus = obs_chunk.clone()
            j = model.names.index(modality)
            obs_plus[:, j] = 1.0                                  # counterfactually ordered
            pres_plus = model.subset_presence(obs_plus, ctx | {modality})
            acc = np.zeros(len(sel))
            with torch.no_grad():
                for s in range(cfg.n_samples):
                    xx = dict(x_chunk)
                    xx[modality] = torch.from_numpy(
                        np.ascontiguousarray(draws[s], dtype=np.float32))
                    p_plus = torch.sigmoid(model(xx, pres_plus)).numpy()
                    acc += _kl_bern(p_plus, p_base)
            delta[sel] = acc / (cfg.n_samples * LN2)

    return PatientGainResult(modality=modality, context=ctx, delta_bits=delta,
                             base_risk=base, donor_distance=dist, observed=mod_obs,
                             n_donors=n_donor_total, sampler=cfg.sampler)


def retrospective_gain(cf, modality: str, context: Iterable[str] | None = None
                       ) -> np.ndarray:
    r"""Realised per-patient PVI gain :math:`\mathrm{pvi}_i(S\cup m)-\mathrm{pvi}_i(S)`.

    Uses the actual test result *and* the actual outcome, so it is a
    retrospective audit quantity, not a bedside score.  Its cohort mean is the
    conditional gain :math:`U(m\mid S)`, which makes it the natural yardstick for
    validating the prospective estimator.
    """
    ctx = frozenset(context) if context is not None else (frozenset(cf.names) - {modality})
    ctx = ctx - {modality}
    return cf.pvi(ctx | {modality}) - cf.pvi(ctx)


def consistency_check(cf, cohort, modality: str, context: Iterable[str] | None = None,
                      cfg: GainConfig | None = None) -> dict:
    r"""Does the prospective score average to the retrospective cohort gain?

    :math:`\mathbb E_i[\Delta_i(m\mid S)]` and
    :math:`\mathbb E_i[\mathrm{pvi}_i(S\cup m)-\mathrm{pvi}_i(S)]` estimate the
    same population quantity by the tower property, through completely different
    routes -- one integrates over hypothetical results without labels, the other
    uses realised results and labels.  Agreement is a strong joint check on the
    sampler, the masked family and the cross-fitting.
    """
    res = patient_expected_gain(cf, cohort, modality, context, cfg)
    retro = retrospective_gain(cf, modality, context)
    obs = res.observed
    return {
        "modality": modality,
        "prospective_mean_bits": float(res.delta_bits.mean()),
        "retrospective_mean_bits": float(retro.mean()),
        "prospective_mean_observed": float(res.delta_bits[obs].mean()) if obs.any() else np.nan,
        "retrospective_mean_observed": float(retro[obs].mean()) if obs.any() else np.nan,
        "abs_difference": float(abs(res.delta_bits.mean() - retro.mean())),
        "spearman": float(pd.Series(res.delta_bits).corr(pd.Series(retro), method="spearman")),
        "n_donors": res.n_donors,
    }


# --------------------------------------------------------------------------- #
# Identification diagnostics
# --------------------------------------------------------------------------- #
def propensity_model(cohort, modality: str, context: Iterable[str],
                     rows: np.ndarray | None = None, seed: int = 0
                     ) -> tuple[np.ndarray, float]:
    """Logistic propensity of having ordered ``modality``, given the context.

    Returns the fitted propensities and their cross-validated AUROC.  A
    propensity model that predicts ordering almost perfectly is a warning, not a
    success: it means there is a region of patient space where the test is never
    ordered and the counterfactual is unidentified there.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from infogain.theory.lemmas import auroc

    rows = np.arange(cohort.n) if rows is None else np.asarray(rows)
    feats = []
    for m in context:
        b = cohort.blocks[m]
        feats.append(b.values[rows])
        feats.append(b.observed[rows][:, None].astype(np.float32))
    X = np.concatenate(feats, axis=1)
    y = cohort.blocks[modality].observed[rows].astype(int)
    if y.min() == y.max():
        return np.full(len(rows), float(y.mean())), float("nan")
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    p = cross_val_predict(pipe, X, y, cv=5, method="predict_proba")[:, 1]
    return p, float(auroc(y, p))


def overlap_diagnostics(cohort, modality: str, context: Iterable[str],
                        rows: np.ndarray | None = None,
                        low: float = 0.02, high: float = 0.98) -> dict:
    """Positivity check for the counterfactual "what if we had ordered it"."""
    p, auc = propensity_model(cohort, modality, context, rows)
    rows = np.arange(cohort.n) if rows is None else np.asarray(rows)
    obs = cohort.blocks[modality].observed[rows]
    return {
        "modality": modality,
        "order_rate": float(obs.mean()),
        "propensity_auroc": auc,
        "frac_propensity_below": float((p < low).mean()),
        "frac_propensity_above": float((p > high).mean()),
        "frac_off_support": float(((p < low) | (p > high)).mean()),
        "propensity_p01": float(np.quantile(p, 0.01)),
        "propensity_p99": float(np.quantile(p, 0.99)),
        "ess_ratio": float((p.sum() ** 2) / (np.sum(p ** 2) * len(p))),
    }
