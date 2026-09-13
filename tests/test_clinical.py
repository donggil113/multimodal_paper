"""Cost model, policies and the reduction study."""
import numpy as np
import pandas as pd
import pytest

from infogain.clinical.cost import CostModel, CostWeights, bits_per_dollar
from infogain.clinical.net_benefit import decision_curve, empirical_net_benefit
from infogain.clinical.policy import (
    evaluate_policy, policy_baseline_only, policy_fixed, policy_infogain,
    policy_order_all, policy_random, policy_risk_band, probs_under_policy,
    reduction_study,
)
from infogain.vinfo.core import Modality, ModalitySpec, subset_key


class _FakeCF:
    """A cross-fitted family stub with hand-set subset probabilities."""

    def __init__(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        self.names = ["vitals", "a", "b"]
        self.baseline = frozenset({"vitals"})
        self.y = (rng.random(n) < 0.2).astype(float)
        sig = rng.normal(size=n) + 1.2 * self.y
        self._p = {}
        for s in [frozenset(), self.baseline, self.baseline | {"a"},
                  self.baseline | {"b"}, self.baseline | {"a", "b"}]:
            strength = 0.2 * len(s)
            self._p[subset_key(s)] = 1 / (1 + np.exp(-(strength * sig - 1.4)))
        self.subsets = [frozenset(k.split("+")) if k != "EMPTY" else frozenset()
                        for k in self._p]
        self.n_seeds = 1

    def p(self, subset, seed_index=None):
        return self._p[subset_key(subset)]

    def has(self, subset):
        return subset_key(subset) in self._p

    def pvi(self, subset, seed_index=None, clip=8.0, rows=None):
        from infogain.vinfo.core import label_logprob, pointwise_v_information
        pm = self.p(subset)
        p0 = np.full_like(pm, self.y.mean())
        return pointwise_v_information(label_logprob(pm, self.y),
                                       label_logprob(p0, self.y), clip=clip)

    def seed_means(self, subset, rows=None):
        return np.array([self.pvi(subset).mean()])


@pytest.fixture
def spec():
    return ModalitySpec([Modality("vitals", 4, always_available=True),
                         Modality("a", 6, cost_usd=25.0, turnaround_min=15.0),
                         Modality("b", 6, cost_usd=100.0, turnaround_min=120.0)])


def test_cost_weights_change_the_ranking(spec):
    money = CostModel(spec, CostWeights.money_only())
    assert money.cost("b") / money.cost("a") == pytest.approx(4.0)
    thr = CostModel(spec, CostWeights.throughput_limited(2.0))
    # once minutes are priced, the slow test becomes relatively more expensive
    assert thr.cost("b") / thr.cost("a") > money.cost("b") / money.cost("a")


def test_subset_cost_ignores_free_modalities(spec):
    cm = CostModel(spec)
    assert cm.subset_cost({"vitals", "a"}) == pytest.approx(25.0)


def test_bits_per_dollar():
    assert np.allclose(bits_per_dollar(np.array([0.1, 0.2]), 10.0), [0.01, 0.02])


def test_probs_under_policy_reads_the_right_subset(spec):
    cf = _FakeCF()
    n = len(cf.y)
    chosen = {"a": np.zeros(n, bool), "b": np.zeros(n, bool)}
    chosen["a"][:100] = True
    chosen["b"][50:150] = True
    p = probs_under_policy(cf, cf.baseline, chosen)
    assert np.allclose(p[:50], cf.p(cf.baseline | {"a"})[:50])
    assert np.allclose(p[50:100], cf.p(cf.baseline | {"a", "b"})[50:100])
    assert np.allclose(p[100:150], cf.p(cf.baseline | {"b"})[100:150])
    assert np.allclose(p[150:], cf.p(cf.baseline)[150:])


def test_order_all_and_baseline_only_are_the_extremes(spec):
    cf = _FakeCF()
    cm = CostModel(spec)
    allp = policy_order_all(cf, cm, ["a", "b"])
    nonep = policy_baseline_only(cf, cm, ["a", "b"])
    assert allp.n_tests.mean() == 2.0 and nonep.n_tests.mean() == 0.0
    assert allp.cost.mean() == pytest.approx(125.0)
    assert nonep.cost.mean() == 0.0


def test_infogain_threshold_monotonically_reduces_testing(spec):
    cf = _FakeCF()
    cm = CostModel(spec)
    rng = np.random.default_rng(0)
    deltas = {"a": np.abs(rng.normal(0.02, 0.02, len(cf.y))),
              "b": np.abs(rng.normal(0.02, 0.02, len(cf.y)))}
    prev = np.inf
    for lam in (0.0, 1e-4, 5e-4, 1e-3, 1e-2):
        r = policy_infogain(cf, cm, deltas, lam)
        assert r.n_tests.mean() <= prev + 1e-9
        prev = r.n_tests.mean()


def test_risk_band_tests_only_the_middle(spec):
    cf = _FakeCF()
    cm = CostModel(spec)
    base = cf.p(cf.baseline)
    r = policy_risk_band(cf, cm, ["a", "b"], base, 0.25, 0.75)
    assert 0.4 < r.chosen["a"].mean() < 0.6
    assert not r.chosen["a"][np.argmin(base)]


def test_evaluate_policy_reports_zero_gap_for_order_all(spec):
    cf = _FakeCF()
    cm = CostModel(spec)
    allp = policy_order_all(cf, cm, ["a", "b"])
    ev = evaluate_policy(allp, cf.y, allp.probs, n_orderable=2)
    assert ev.info_vs_all_bits == pytest.approx(0.0, abs=1e-9)
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in ev.certified_nb_loss.values())


def test_certified_loss_dominates_observed_loss(spec):
    cf = _FakeCF()
    cm = CostModel(spec)
    ref = policy_order_all(cf, cm, ["a", "b"]).probs
    r = policy_baseline_only(cf, cm, ["a", "b"])
    ev = evaluate_policy(r, cf.y, ref, thresholds=(0.05, 0.1, 0.2), n_orderable=2)
    for t in (0.05, 0.1, 0.2):
        observed_loss = (float(empirical_net_benefit(cf.y, ref, np.array([t]))[0])
                         - ev.net_benefit[t])
        assert ev.certified_nb_loss[t] >= observed_loss - 1e-9


def test_reduction_study_produces_a_frontier(spec):
    cf = _FakeCF(n=600, seed=2)
    cm = CostModel(spec)
    rng = np.random.default_rng(1)
    deltas = {"a": np.abs(rng.normal(0.03, 0.03, len(cf.y))),
              "b": np.abs(rng.normal(0.01, 0.02, len(cf.y)))}
    study = reduction_study(cf, None, deltas, cm, n_lambda=8)
    assert len(study.frontier) >= 3
    assert study.frontier["test_fraction"].is_monotonic_increasing
    assert {"order_all", "baseline_only"} <= set(study.comparators["policy"])
    assert isinstance(study.summary(), str)


def test_decision_curve_beats_nothing_at_sensible_thresholds():
    rng = np.random.default_rng(0)
    n = 20000
    p = rng.beta(2, 12, n)
    y = (rng.random(n) < p).astype(float)
    dc = decision_curve(y, p, np.linspace(0.02, 0.4, 20))
    assert (dc.net_benefit[:5] > 0).all()
    assert (dc.net_benefit >= dc.treat_all - 1e-9).all()
