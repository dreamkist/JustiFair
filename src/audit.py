"""Auditing module for legal AI decision systems."""

from typing import Any, Dict
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    count
)

class LegalBiasAuditor:
    """Disaggregated fairness auditor tailored for judicial and bail decision support."""

    def __init__(self, metrics: Dict[str, Any] = None):
        self.metrics = metrics or {
            "Sample Count": count,
            "Accuracy": accuracy_score,
            "Selection Rate (High Risk %)": selection_rate,
            "False Positive Rate (Unjust Risk Flag)": false_positive_rate,
            "False Negative Rate (Overlooked Risk)": false_negative_rate,
            "Precision (PPV)": precision_score,
        }

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        sensitive_features: pd.Series
    ) -> MetricFrame:
        """Run MetricFrame evaluation across demographic groups."""
        return MetricFrame(
            metrics=self.metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
