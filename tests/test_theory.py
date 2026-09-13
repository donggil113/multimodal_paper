"""The paper's mathematical claims, as executable assertions."""
import numpy as np
import pytest

from infogain.clinical.net_benefit import (
    bayes_value, check_identity, empirical_net_benefit, information_gain_nats,
    min_detectable_gain, model_net_benefit, nest_calibrate, restricted_information,
    safe_omission_bound_global, safe_omission_bound_local,
)
from infogain.theory.lemmas import (
    achievable_auroc_bracket, auroc, auroc_hull, best_stratified_bound, entropy_bits,
    ks_signed, stratified_auroc_gain_bound, verify_js_tv, verify_ks_auroc,
)
from infogain.theory.verify import (
    verify_pvi_consistency, verify_safe_omission, verify_schervish_identity,
    verify_theorem1,
)

LN2 = float(np.log(2.0))


def test_lemma_js_tv():
    up, lo, sharp = verify_js_tv(n_trials=4000, seed=3)
    assert up.holds, f"I <= H(pi)TV violated {up.violated} times"
    assert lo.holds, f"proved lower bound violated {lo.violated} times"
    # the sharper constant is not a theorem; we only report that it also held
    assert sharp.violated == 0


def test_lemma_ks_auroc_sandwich():
    lower, upper = verify_ks_auroc(n_trials=600, n=800, seed=5)
    assert lower.holds and upper.holds


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.3).astype(int)
    s = rng.normal(size=2000) + 0.8 * y
    assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-9
    assert auroc_hull(y, s) >= auroc(y, s) - 1e-9


def test_auroc_handles_heavy_ties():
    y = np.array([1, 1, 0, 0])
    s = np.array([1.0, 1.0, 1.0, 1.0])
    assert auroc(y, s) == pytest.approx(0.5)


def test_schervish_identity_exact():
    rep = verify_schervish_identity(n_trials=12, n=20000, seed=1)
    assert rep.holds
    assert abs(rep.worst_slack) < 1e-3


def test_identity_on_a_concrete_pair():
    rng = np.random.default_rng(7)
    z = rng.normal(size=60000)
    eta2 = 1 / (1 + np.exp(-(z - 2.0 + rng.normal(0, 1.1, 60000))))
    eta1 = nest_calibrate(eta2, 1 / (1 + np.exp(-(z - 2.0))), n_bins=500)
    chk = check_identity(eta2, eta1)
    assert chk.rel_error < 1e-3
    # the restricted version can only be smaller than the total
    assert 0 < restricted_information(eta2, eta1, 0.02, 0.30) <= chk.kl_nats / LN2 + 1e-9


def test_safe_omission_bounds_dominate():
    g, l = verify_safe_omission(n_trials=40, n=15000, seed=2)
    assert g.holds and l.holds


def test_local_bound_tighter_than_global():
    for t in (0.02, 0.05, 0.1, 0.3):
        for bits in (0.001, 0.01, 0.1):
            assert safe_omission_bound_local(bits, t) <= safe_omission_bound_global(bits, t) + 1e-12


def test_min_detectable_gain_is_an_inverse():
    for t in (0.05, 0.2):
        bits = min_detectable_gain(t, 0.002)
        assert safe_omission_bound_local(bits, t) == pytest.approx(0.002, rel=1e-3)


def test_theorem1_bound_never_exceeds_achievable():
    a, identity, informative = verify_theorem1(n_trials=25, n=6000, seed=4)
    assert a.holds
    # the theorem's content is an identity with the witness score's AUROC gain;
    # it should hold to machine precision, not merely as an inequality
    assert identity.holds and abs(identity.worst_slack) < 1e-9
    assert informative.worst_slack > 0.3, "bound is vacuous on synergy problems"


def test_theorem1_zero_for_uninformative_addition():
    rng = np.random.default_rng(11)
    n = 8000
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(1.2 * x - 1.5)))
    y = (rng.random(n) < p).astype(int)
    b = stratified_auroc_gain_bound(y, p, p, n_bins=8)
    assert abs(b.bound) < 1e-9


def test_pvi_recovers_closed_form_information():
    rep = verify_pvi_consistency(n=200000, seed=0)
    assert rep.holds


def test_bracket_contains_truth_on_a_gaussian_problem():
    rng = np.random.default_rng(0)
    n = 200000
    y = (rng.random(n) < 0.2).astype(int)
    x = rng.normal(size=n) + 1.1 * y
    # exact posterior for the Gaussian location model
    pi = 0.2
    lr = np.exp(1.1 * x - 0.5 * 1.1 ** 2)
    eta = pi * lr / (pi * lr + (1 - pi))
    h = lambda q: -(q * np.log2(q) + (1 - q) * np.log2(1 - q))  # noqa: E731
    info = entropy_bits(pi) - float(np.mean(h(np.clip(eta, 1e-12, 1 - 1e-12))))
    lo, hi = achievable_auroc_bracket(info, pi)
    a = auroc(y, eta)
    assert lo - 1e-6 <= a <= hi + 1e-6


def test_net_benefit_model_and_empirical_agree_when_calibrated():
    rng = np.random.default_rng(3)
    n = 200000
    p = rng.beta(1.5, 12, size=n)
    y = (rng.random(n) < p).astype(float)
    t = np.linspace(0.02, 0.4, 15)
    assert np.max(np.abs(empirical_net_benefit(y, p, t) - model_net_benefit(p, t))) < 5e-3


def test_bayes_value_matches_bruteforce():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    t = np.array([0.1, 0.5, 0.9])
    brute = np.array([np.mean(np.maximum(p - tt, 0)) for tt in t])
    assert np.allclose(bayes_value(p, t), brute)
