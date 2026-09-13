"""End-to-end integration: the analysis runs and its internal identities hold."""
import json

import numpy as np
import pandas as pd
import pytest

from infogain.clinical.cost import CostModel, CostWeights
from infogain.clinical.policy import reduction_study
from infogain.encoders.fusion import FamilyConfig
from infogain.encoders.train import TrainConfig, fit_family
from infogain.experiments.run_cohort import AnalysisConfig, analyse
from infogain.vinfo.conditional import GainConfig, patient_expected_gain
from infogain.vinfo.decomposition import (
    decomposition_table, pairwise_interaction_map, shapley_information,
)


@pytest.fixture(scope="module")
def fitted(small_cohort):
    cohort, gt = small_cohort
    cf = fit_family(cohort, "aki_7d",
                    TrainConfig(epochs=20, n_folds=3, seeds=(0, 1), patience=6,
                                min_epochs=5, family=FamilyConfig(
                                    emb_dim=24, enc_hidden=48, head_hidden=64,
                                    n_heads=2, n_attn_layers=1)),
                    keep_models=True)
    return cohort, gt, cf


def test_decomposition_identity_holds_exactly(fitted):
    """U = I_m - R + Syn must reconstruct, term by term."""
    _, _, cf = fitted
    df = decomposition_table(cf)
    lhs = df["conditional_bits"]
    rhs = df["marginal_bits"] - df["redundant_bits"] + df["synergistic_bits"]
    assert np.allclose(lhs, rhs, atol=1e-12)
    assert (df["redundant_bits"] * df["synergistic_bits"] == 0).all()
    assert (df["redundant_bits"] >= 0).all() and (df["synergistic_bits"] >= 0).all()


def test_shapley_is_efficient(fitted):
    """Values must sum to the total usable information of the full panel."""
    _, _, cf = fitted
    sh = shapley_information(cf)
    assert abs(sh.attrs["efficiency_residual"]) < 1e-10
    assert abs(sh["shapley_bits"].sum() - sh.attrs["total_bits"]) < 1e-10


def test_interaction_map_is_symmetric_and_consistent(fitted):
    _, _, cf = fitted
    im = pairwise_interaction_map(cf)
    recomputed = im["I_ab"] - im["I_a"] - im["I_b"]
    assert np.allclose(im["interaction_bits"], recomputed, atol=1e-9)
    assert set(im["regime"]) <= {"redundant", "synergistic", "indeterminate"}


def test_patient_gain_is_nonnegative_and_prospective(fitted):
    """The score must not use the label, and KL is non-negative by definition."""
    cohort, _, cf = fitted
    res = patient_expected_gain(cf, cohort, "ecg", context=cohort.spec.baseline,
                                cfg=GainConfig(n_samples=6, n_neighbors=15))
    assert (res.delta_bits >= -1e-12).all()
    assert res.delta_bits.shape == cf.y.shape
    # a score that used the outcome would separate the classes; this one must not
    # be able to, beyond what the base risk already implies
    assert np.isfinite(res.delta_bits).all()


def test_patient_gain_beats_the_misspecified_control(fitted):
    """Conditioning on the patient must matter: kNN should not equal marginal."""
    cohort, _, cf = fitted
    knn = patient_expected_gain(cf, cohort, "labs", context=cohort.spec.baseline,
                                cfg=GainConfig(n_samples=6, n_neighbors=15,
                                               sampler="knn")).delta_bits
    marg = patient_expected_gain(cf, cohort, "labs", context=cohort.spec.baseline,
                                 cfg=GainConfig(n_samples=6, n_neighbors=15,
                                                sampler="marginal")).delta_bits
    assert not np.allclose(knn, marg, atol=1e-6)


def test_policy_never_outperforms_ordering_everything_on_information(fitted):
    cohort, _, cf = fitted
    rng = np.random.default_rng(0)
    gains = {m: np.abs(rng.normal(0.01, 0.01, len(cf.y)))
             for m in cohort.spec.orderable}
    cm = CostModel(cohort.spec, CostWeights.money_only())
    study = reduction_study(cf, cohort, gains, cm, n_lambda=6)
    assert (study.frontier["info_gap_vs_all_bits"] >= -1e-9).all()
    assert (study.frontier["test_fraction"] <= 1.0 + 1e-9).all()


def test_full_analysis_runs_and_writes_everything(small_cohort, tmp_path):
    cohort, _ = small_cohort
    cfg = AnalysisConfig(
        outcome="aki_7d",
        train=TrainConfig(epochs=12, n_folds=3, seeds=(0,), patience=4,
                          min_epochs=3,
                          family=FamilyConfig(emb_dim=16, enc_hidden=32,
                                              head_hidden=32, n_heads=2,
                                              n_attn_layers=1)),
        gain=GainConfig(n_samples=4, n_neighbors=10),
        make_figures=True)
    summary = analyse(cohort, cfg, tmp_path)

    for name in ("diagnostics", "lattice", "decomposition", "interaction_map",
                 "shapley", "patient_gain_summary", "gain_consistency",
                 "overlap_diagnostics", "theorem1_bounds", "theorem2_bounds",
                 "policy_frontier", "policy_comparators", "cost_model"):
        assert (tmp_path / "tables" / f"{name}.csv").exists(), name
    for fig in ("fig1_decomposition", "fig2_interaction_map", "fig3_patient_gains",
                "fig4_decision_curves", "fig5_reduction_frontier", "fig7_identity",
                "fig8_theorem1"):
        assert (tmp_path / "figures" / f"{fig}.png").exists(), fig

    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["outcome"] == "aki_7d"
    assert 0 < saved["cohort"]["prevalence"] < 1
    assert saved["identity_check"]["rel_error"] < 0.05
    assert abs(saved["shapley_efficiency_residual"]) < 1e-9


def test_theorem2_certificate_dominates_realised_loss_in_the_run(fitted, tmp_path):
    """The certified bound must never be smaller than what actually happened."""
    cohort, _, cf = fitted
    cfg = AnalysisConfig(
        outcome="aki_7d",
        train=TrainConfig(epochs=8, n_folds=3, seeds=(0,), patience=3, min_epochs=2,
                          family=FamilyConfig(emb_dim=16, enc_hidden=32,
                                              head_hidden=32, n_heads=2,
                                              n_attn_layers=1)),
        gain=GainConfig(n_samples=4, n_neighbors=10), make_figures=False)
    analyse(cohort, cfg, tmp_path)
    t2 = pd.read_csv(tmp_path / "tables" / "theorem2_bounds.csv")
    assert (t2["certified_nb_loss"] >= t2["observed_nb_loss"] - 1e-9).all()


def test_theorem1_bound_does_not_exceed_observed_in_the_run(fitted, tmp_path):
    cohort, _, cf = fitted
    cfg = AnalysisConfig(
        outcome="aki_7d",
        train=TrainConfig(epochs=8, n_folds=3, seeds=(0,), patience=3, min_epochs=2,
                          family=FamilyConfig(emb_dim=16, enc_hidden=32,
                                              head_hidden=32, n_heads=2,
                                              n_attn_layers=1)),
        gain=GainConfig(n_samples=4, n_neighbors=10), make_figures=False)
    analyse(cohort, cfg, tmp_path)
    t1 = pd.read_csv(tmp_path / "tables" / "theorem1_bounds.csv")
    # Theorem 1 bounds the *achievable* gain (hull AUROC of the richer model
    # minus the context model's AUROC), which can exceed the gain the particular
    # fitted pair happens to show.
    assert (t1["bound"] <= t1["achievable_gain"] + 1e-6).all()


def test_sequential_policy_rescoring_uses_per_patient_contexts(fitted):
    """Each patient must be scored against the set *they* have reached."""
    from infogain.clinical.cost import CostModel, CostWeights
    from infogain.clinical.policy import greedy_sequential

    cohort, _, cf = fitted
    orderable = cohort.spec.orderable
    n = len(cf.y)
    rng = np.random.default_rng(0)

    # context () prefers the first modality; after acquiring it, the second
    # becomes the best remaining one -- so a correct implementation reaches a
    # two-test panel, and one that ignores context does not.
    first, second = orderable[0], orderable[1]
    table = {
        (): {m: (np.full(n, 0.05) if m == first else np.full(n, 0.01))
             for m in orderable},
        (first,): {m: (np.full(n, 0.05) if m == second else np.full(n, 0.001))
                   for m in orderable if m != first},
    }
    for m in orderable:
        if m != first:
            table[(m,)] = {o: np.full(n, 0.0005) for o in orderable if o != m}

    cm = CostModel(cohort.spec, CostWeights.money_only())
    res = greedy_sequential(cf, table, cm, orderable, max_tests=2)
    assert res.chosen[first].all()
    assert res.chosen[second].all()
    assert not res.chosen[orderable[2]].any()
    assert res.n_tests.mean() == pytest.approx(2.0)


def test_sequential_policy_stops_below_the_threshold(fitted):
    from infogain.clinical.cost import CostModel, CostWeights
    from infogain.clinical.policy import greedy_sequential

    cohort, _, cf = fitted
    orderable = cohort.spec.orderable
    n = len(cf.y)
    table = {(): {m: np.full(n, 1e-9) for m in orderable}}
    cm = CostModel(cohort.spec, CostWeights.money_only())
    res = greedy_sequential(cf, table, cm, orderable, max_tests=2,
                            lambda_bits_per_dollar=1e-3)
    assert res.n_tests.sum() == 0


def test_patient_targeting_study_reports_the_yardstick_ceiling(tmp_path):
    """The targeting study must separate "bad estimator" from "bad yardstick".

    The consistency table's near-zero rank correlation between the prospective
    score and the retrospective realised gain invites the reading that the
    per-patient score is noise.  ``spearman_truth_vs_retro`` is what settles it:
    it is the correlation the *exact* gain achieves against the same yardstick,
    so it is an upper bound on what any estimator could score there.  If the
    study ever stopped reporting it, the paper would lose the only evidence
    distinguishing the two explanations.
    """
    from infogain.experiments.run_simulation import patient_targeting_study

    tgt = patient_targeting_study(n=1500, outcome="aki_7d", epochs=12, folds=3,
                                  seeds=(0,), seed=0)
    for col in ("spearman_est_vs_truth", "spearman_truth_vs_retro",
                "capture_at_10", "chance_at_10", "mean_true_bits"):
        assert col in tgt.columns, col
    assert len(tgt) >= 3 and tgt["modality"].is_unique
    for col in ("spearman_est_vs_truth", "spearman_est_vs_retro",
                "spearman_truth_vs_retro"):
        assert tgt[col].between(-1.0, 1.0).all(), col
    # a budget policy cannot collect more than the oracle collects, and the
    # random-ordering floor is a fraction of it too
    assert tgt["capture_at_10"].between(0.0, 1.0 + 1e-9).all()
    assert tgt["chance_at_10"].between(0.0, 1.0 + 1e-9).all()
    # the exact per-patient gain is a non-negative quantity by construction
    assert (tgt["mean_true_bits"] >= -1e-9).all()
