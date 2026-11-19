"""Tests for JustiFair modules."""

import numpy as np
import pandas as pd
from src.audit import LegalBiasAuditor
from src.mitigation import FairLegalClassifier

def test_legal_bias_auditor():
    y_true = pd.Series([1, 0, 1, 0, 1, 0, 0, 1])
    y_pred = pd.Series([1, 0, 0, 0, 1, 1, 0, 1])
    sensitive = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])

    auditor = LegalBiasAuditor()
    frame = auditor.evaluate(y_true, y_pred, sensitive)
    assert "Accuracy" in frame.by_group.columns
    assert "False Positive Rate (Unjust Risk Flag)" in frame.by_group.columns
    assert frame.difference()["Accuracy"] >= 0

def test_fair_legal_classifier():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(80, 3), columns=["f1", "f2", "f3"])
    sensitive = pd.Series(np.random.choice(["A", "B"], size=80))
    y = pd.Series(np.random.choice([0, 1], size=80))

    model = FairLegalClassifier()
    model.fit(X, y, sensitive_features=sensitive)
    preds = model.predict(X)
    assert len(preds) == len(y)
