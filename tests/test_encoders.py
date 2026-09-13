"""The predictive family: masking closure, monotone behaviour, cross-fitting."""
import numpy as np
import pytest
import torch

from infogain.encoders.fusion import (
    FamilyConfig, MaskedFusionFamily, MaskSampler, Standardizer, pack,
)
from infogain.encoders.train import TrainConfig, default_subsets, fit_family
from infogain.vinfo.core import subset_key


@pytest.fixture(scope="module")
def tiny_family():
    torch.manual_seed(0)
    m = MaskedFusionFamily({"a": 5, "b": 7, "c": 3},
                           FamilyConfig(emb_dim=8, enc_hidden=12, head_hidden=16,
                                        n_heads=2, n_attn_layers=1))
    # The head's last layer is zero-initialised so the empty-mask member starts
    # at the prevalence; leaving it that way would make every closure test below
    # pass trivially on a constant output. Randomise it, and switch off dropout,
    # so the tests actually exercise the masking path.
    torch.nn.init.normal_(m.head[-1].weight, std=0.5)
    torch.nn.init.normal_(m.head[-1].bias, std=0.5)
    m.eval()
    return m


def test_masking_closure_absent_modality_cannot_influence_output(tiny_family):
    """Masking a modality must make its features irrelevant -- exactly, not roughly.

    This is the assumption behind monotone information on the subset lattice.  If
    a masked modality leaked even slightly, every conditional gain would be
    contaminated.
    """
    x = {"a": torch.randn(6, 5), "b": torch.randn(6, 7), "c": torch.randn(6, 3)}
    present = torch.tensor([[1.0, 0.0, 1.0]] * 6)
    out1 = tiny_family(x, present)
    x2 = dict(x)
    x2["b"] = torch.randn(6, 7) * 100      # wildly different masked input
    out2 = tiny_family(x2, present)
    assert torch.allclose(out1, out2, atol=1e-6)
    # guard against a vacuous test: a *present* modality must matter
    x3 = dict(x)
    x3["a"] = torch.randn(6, 5) * 100
    assert not torch.allclose(out1, tiny_family(x3, present), atol=1e-4)


def test_embedding_also_respects_masking(tiny_family):
    x = {"a": torch.randn(4, 5), "b": torch.randn(4, 7), "c": torch.randn(4, 3)}
    present = torch.tensor([[1.0, 0.0, 0.0]] * 4)
    e1 = tiny_family.embed(x, present)
    x2 = dict(x); x2["c"] = torch.randn(4, 3) * 50
    assert torch.allclose(e1, tiny_family.embed(x2, present), atol=1e-6)
    x3 = dict(x); x3["a"] = torch.randn(4, 5) * 50
    assert not torch.allclose(e1, tiny_family.embed(x3, present), atol=1e-4)


def test_empty_mask_gives_a_constant_prediction(tiny_family):
    x = {"a": torch.randn(9, 5), "b": torch.randn(9, 7), "c": torch.randn(9, 3)}
    out = tiny_family(x, torch.zeros(9, 3))
    assert torch.allclose(out, out[0].expand_as(out), atol=1e-6)


def test_concat_fusion_also_closes_under_masking():
    torch.manual_seed(1)
    m = MaskedFusionFamily({"a": 4, "b": 6},
                           FamilyConfig(emb_dim=8, enc_hidden=10, head_hidden=10,
                                        fusion="concat"))
    torch.nn.init.normal_(m.head[-1].weight, std=0.5)
    m.eval()
    x = {"a": torch.randn(5, 4), "b": torch.randn(5, 6)}
    p = torch.tensor([[1.0, 0.0]] * 5)
    x2 = dict(x); x2["b"] = torch.randn(5, 6) * 30
    assert torch.allclose(m(x, p), m(x2, p), atol=1e-6)
    x3 = dict(x); x3["a"] = torch.randn(5, 4) * 30
    assert not torch.allclose(m(x, p), m(x3, p), atol=1e-4)


def test_subset_presence_intersects_with_acquisition(tiny_family):
    observed = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    pres = tiny_family.subset_presence(observed, {"a", "c"})
    assert pres.tolist() == [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]


def test_mask_sampler_covers_the_lattice_corners():
    s = MaskSampler(["a", "b", "c"], frozenset({"a"}))
    g = torch.Generator().manual_seed(0)
    m = s.sample(6000, g)
    assert (m.sum(1) == 0).float().mean() > 0.03      # empty set is trained
    assert (m.sum(1) == 3).float().mean() > 0.15      # full set is trained
    assert m[:, 0].mean() > m[:, 1].mean()            # baseline kept more often


def test_standardizer_uses_only_observed_rows():
    vals = np.zeros((100, 2), dtype=np.float32)
    vals[:50] = 10.0
    obs = np.zeros(100, bool)
    obs[:50] = True
    std = Standardizer().fit({"m": vals}, {"m": obs}, np.arange(100))
    assert np.allclose(std.mean["m"], 10.0)           # the zero-filled rows are ignored
    out = std.transform({"m": vals}, {"m": obs})["m"]
    assert np.allclose(out[50:], 0.0)


def test_default_subsets_include_what_the_decomposition_needs():
    subs = {subset_key(s) for s in default_subsets(["v", "a", "b"], {"v"})}
    assert "EMPTY" in subs and "v" in subs and "a+v" in subs and "a+b+v" in subs
    assert "a" in subs and "b" in subs                 # stand-alone value of each test


def test_fit_family_produces_out_of_fold_predictions(small_cohort):
    cohort, _ = small_cohort
    cf = fit_family(cohort, "aki_7d",
                    TrainConfig(epochs=6, n_folds=3, seeds=(0,), min_epochs=2,
                                patience=3))
    n = len(cf.y)
    assert set(np.unique(cf.fold_id)) == {0, 1, 2}
    assert cf.p(cf.baseline).shape == (n,)
    assert np.all((cf.p(cf.baseline) > 0) & (cf.p(cf.baseline) < 1))
    # every patient is scored by a model trained without them
    assert np.all(np.isfinite(cf.pvi(cf.baseline)))


def test_null_anchor_is_the_fold_prevalence(small_cohort):
    cohort, _ = small_cohort
    cf = fit_family(cohort, "aki_7d",
                    TrainConfig(epochs=4, n_folds=2, seeds=(0,), min_epochs=1, patience=2))
    for k in (0, 1):
        sel = cf.fold_id == k
        others = cf.y[~sel].mean()
        assert cf.p_null[sel][0] == pytest.approx(others, abs=1e-6)


def test_pvi_of_the_empty_subset_is_near_zero(small_cohort):
    cohort, _ = small_cohort
    cf = fit_family(cohort, "aki_7d",
                    TrainConfig(epochs=4, n_folds=2, seeds=(0,), min_epochs=1, patience=2))
    # the empty-mask member and the constant null differ only by training noise
    assert abs(cf.null_gap_bits()) < 0.05


def test_keep_models_retains_one_model_per_fold_and_seed(small_cohort):
    cohort, _ = small_cohort
    cf = fit_family(cohort, "aki_7d",
                    TrainConfig(epochs=3, n_folds=2, seeds=(0, 1), min_epochs=1,
                                patience=2), keep_models=True)
    assert len(cf.fold_models) == 2 and len(cf.fold_models[0]) == 2
    assert len(cf.standardizers) == 2
