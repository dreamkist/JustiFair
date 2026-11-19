"""Mitigation pipelines for legal decision support."""

from typing import Optional
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds, DemographicParity

class FairLegalClassifier:
    """Fairness-constrained classifier training without test-time sensitive feature reliance."""

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        constraint: str = "equalized_odds",
        eps: float = 0.01
    ):
        estimator = base_estimator or LogisticRegression(max_iter=1000)
        fairness_constraint = (
            EqualizedOdds() if constraint == "equalized_odds" else DemographicParity()
        )
        self.mitigator = ExponentiatedGradient(
            estimator=estimator,
            constraints=fairness_constraint,
            eps=eps
        )

    def fit(self, X, y, sensitive_features):
        self.mitigator.fit(X, y, sensitive_features=sensitive_features)
        return self

    def predict(self, X):
        return self.mitigator.predict(X)
