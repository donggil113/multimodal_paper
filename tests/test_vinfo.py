"""Estimator algebra: PVI, intervals, decomposition identities, Shapley axioms."""
import numpy as np
import pytest

from infogain.vinfo.bounds import (
    bootstrap_ci, empirical_bernstein_ci, hoeffding_ci, mlp_rademacher,
    paired_permutation_pvalue, two_sided_bound,
)
from infogain.vinfo.core import (
    Modality,
    ModalitySpec,
    iter_subsets,
    label_logprob,
    paired_gain,
    parse_key,
    pointwise_v_information,
    subset_key,
    v_info_from_pvi,
)
from infogain.vinfo.decomposition import monotone_projection
from infogain.vinfo.estimators import (
    calibration_error, extrapolate_infinite_data, variance_decomposition,
)


def test_subset_keys_round_trip():
    for s in iter_subsets(["a", "b", "c"]):
        assert parse_key(subset_key(s)) == s


def test_modality_spec_roles():
    spec = ModalitySpec([Modality("vitals", 4, always_available=True),
                         Modality("ecg", 8, cost_usd=25.0)])
    assert spec.baseline == frozenset({"vitals"})
    assert spec.orderable == ["ecg"]
    assert spec["ecg"].cost_usd == 25.0
    with pytest.raises(ValueError):
        ModalitySpec([Modality("a", 1), Modality("a", 2)])


def test_pvi_is_zero_for_the_null_model():
    y = np.array([0, 1, 1, 0])
    p = np.full(4, 0.5)
    pvi = pointwise_v_information(label_logprob(p, y), label_logprob(p, y))
    assert np.allclose(pvi, 0.0)


def test_pvi_equals_one_bit_for_a_perfect_binary_predictor():
    y = np.array([0, 1, 0, 1])
    perfect = np.where(y == 1, 1 - 1e-9, 1e-9)
    null = np.full(4, 0.5)
    pvi = pointwise_v_information(label_logprob(perfect, y), label_logprob(null, y))
    assert np.allclose(pvi, 1.0, atol=1e-6)


def test_bernstein_is_valid_and_looser_than_bootstrap():
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 0.5, 4000)
    b_lo, b_hi = empirical_bernstein_ci(x, rng_width=16.0)
    boot_lo, boot_hi = bootstrap_ci(x)
    assert b_lo < x.mean() < b_hi
    assert (b_hi - b_lo) > (boot_hi - boot_lo)
    h_lo, h_hi = hoeffding_ci(x, rng_width=16.0)
    assert (h_hi - h_lo) > (b_hi - b_lo)  # Bernstein adapts to the low variance


def test_bernstein_coverage():
    """Nominal 95% interval should cover well above 95% of the time (it is conservative)."""
    rng = np.random.default_rng(1)
    cover = 0
    for _ in range(300):
        x = rng.normal(0.1, 1.0, 800)
        lo, hi = empirical_bernstein_ci(x, rng_width=16.0)
        cover += int(lo <= 0.1 <= hi)
    assert cover / 300 >= 0.95


def test_paired_gain_is_tighter_than_unpaired():
    rng = np.random.default_rng(2)
    common = rng.normal(0, 3.0, 5000)      # large shared patient-level variance
    a = common + rng.normal(0.05, 0.2, 5000)
    b = common + rng.normal(0.0, 0.2, 5000)
    paired = paired_gain(a, b)
    unpaired_se = np.sqrt(a.var(ddof=1) / 5000 + b.var(ddof=1) / 5000)
    assert paired.se < unpaired_se / 5


def test_permutation_pvalue_calibrated_under_the_null():
    rng = np.random.default_rng(3)
    ps = [paired_permutation_pvalue(rng.normal(0, 1, 400), np.zeros(400),
                                    n_perm=400, seed=i) for i in range(200)]
    assert 0.02 <= np.mean(np.array(ps) < 0.05) <= 0.12


def test_seed_variance_widens_the_interval():
    rng = np.random.default_rng(4)
    pvi = rng.normal(0.05, 0.3, 3000)
    narrow = v_info_from_pvi(pvi)
    wide = v_info_from_pvi(pvi, seed_values=[0.02, 0.05, 0.09, 0.01])
    assert (wide.ci_hi - wide.ci_lo) > (narrow.ci_hi - narrow.ci_lo)
    assert wide.var_seed > 0


def test_variance_decomposition_sums():
    rng = np.random.default_rng(5)
    arr = rng.normal(0.1, 1.0, (4, 2000))
    vd = variance_decomposition(arr)
    assert vd.total == pytest.approx(vd.sampling + vd.seed)
    assert 0 <= vd.seed_share <= 1


def test_capacity_bound_grows_with_weights():
    small = mlp_rademacher([1.0, 1.0], 2.0, 1000)
    big = mlp_rademacher([4.0, 4.0], 2.0, 1000)
    assert big.rademacher > small.rademacher
    assert mlp_rademacher([1.0, 1.0], 2.0, 100000).rademacher < small.rademacher


def test_two_sided_bound_brackets_the_estimate():
    rng = np.random.default_rng(6)
    pvi = rng.normal(0.08, 0.4, 5000)
    tb = two_sided_bound(pvi, mlp_rademacher([1.2, 1.0], 2.0, 5000))
    assert tb.lower < tb.estimate < tb.upper


def test_monotone_projection_is_idempotent_and_closest():
    from itertools import combinations

    names = list("abc")
    subs = [frozenset(c) for r in range(4) for c in combinations(names, r)]
    rng = np.random.default_rng(7)
    vals = {subset_key(s): 0.1 * len(s) + rng.normal(0, 0.03) for s in subs}
    proj = monotone_projection(vals, subs)
    again = monotone_projection(proj, subs)
    assert all(abs(proj[k] - again[k]) < 1e-8 for k in proj)
    for s in subs:
        for t in subs:
            if s < t:
                assert proj[subset_key(s)] <= proj[subset_key(t)] + 1e-8


def test_extrapolation_recovers_a_known_limit():
    m = np.array([1000, 3000, 9000, 27000, 81000], float)
    v = 0.15 - 0.6 * m ** -0.45
    ex = extrapolate_infinite_data(m, v, weights=np.full(5, 1e-5))
    assert ex.i_infinity == pytest.approx(0.15, abs=2e-3)
    assert ex.recovered_fraction < 1.0


def test_calibration_error_zero_for_calibrated():
    rng = np.random.default_rng(8)
    p = rng.beta(2, 8, 200000)
    y = (rng.random(200000) < p).astype(float)
    assert calibration_error(y, p)["ece"] < 5e-3
