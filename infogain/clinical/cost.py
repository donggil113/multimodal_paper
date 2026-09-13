r"""Cost of ordering a test, on the axes that actually bind.

Money is the axis health-economics reviewers ask about, but it is rarely the one
that constrains a clinical pathway: an echocardiogram is expensive *and* slow,
a chest radiograph is cheap but irradiates, a laboratory panel is cheap and fast
but the phlebotomy still has to happen.  Collapsing these to one number early
hides the fact that the optimal policy changes with the setting, so the cost
model keeps them separate and combines them only at the point of use, with
weights the runners sweep over.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from infogain.vinfo.core import ModalitySpec


@dataclass
class CostWeights:
    """Relative price of a dollar, a minute and a unit of invasiveness."""

    money: float = 1.0
    time: float = 0.0        # $ per minute of turnaround
    invasiveness: float = 0.0  # $-equivalent per unit of harm/discomfort

    @classmethod
    def money_only(cls) -> "CostWeights":
        return cls(1.0, 0.0, 0.0)

    @classmethod
    def throughput_limited(cls, dollars_per_minute: float = 2.0) -> "CostWeights":
        """An ED where the binding constraint is length of stay, not price."""
        return cls(1.0, dollars_per_minute, 0.0)

    @classmethod
    def harm_averse(cls, dollars_per_harm: float = 500.0) -> "CostWeights":
        return cls(1.0, 0.0, dollars_per_harm)


@dataclass
class CostModel:
    spec: ModalitySpec
    weights: CostWeights = field(default_factory=CostWeights)
    overrides: dict[str, float] = field(default_factory=dict)

    def cost(self, modality: str) -> float:
        if modality in self.overrides:
            return float(self.overrides[modality])
        m = self.spec[modality]
        return float(self.weights.money * m.cost_usd
                     + self.weights.time * m.turnaround_min
                     + self.weights.invasiveness * m.invasiveness)

    def costs(self) -> dict[str, float]:
        return {m.name: self.cost(m.name) for m in self.spec}

    def subset_cost(self, subset) -> float:
        return float(sum(self.cost(m) for m in subset
                         if not self.spec[m].always_available))

    def table(self) -> pd.DataFrame:
        rows = []
        for m in self.spec:
            rows.append({"modality": m.name, "cost_usd": m.cost_usd,
                         "turnaround_min": m.turnaround_min,
                         "invasiveness": m.invasiveness,
                         "effective_cost": self.cost(m.name),
                         "orderable": not m.always_available})
        return pd.DataFrame(rows)


def bits_per_dollar(delta_bits: np.ndarray, cost: float) -> np.ndarray:
    """Information yield per unit cost -- the quantity the greedy policy ranks on."""
    return np.asarray(delta_bits, dtype=np.float64) / max(cost, 1e-9)


def willingness_to_pay_threshold(nb_per_dollar: float, prevalence: float) -> float:
    r"""Convert a willingness-to-pay into a net-benefit threshold.

    ``nb_per_dollar`` is how much net benefit (in true-positives-per-patient
    units) a payer will buy for one dollar; multiplying by a test's cost gives
    the net-benefit gain the test must deliver to be worth ordering.
    """
    return float(nb_per_dollar * max(prevalence, 1e-12))
