"""CLI runner for JustiFair legal fairness evaluation."""

import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_compas_data
from src.audit import LegalBiasAuditor

def main():
    print("=== JustiFair: Auditing Commercial Legal Decision Support (COMPAS) ===")
    df = load_compas_data()

    y_actual = df["two_year_recid"]
    y_ai = df["score_text"].apply(lambda s: 1 if s in ["Medium", "High"] else 0)
    race = df["race"]

    auditor = LegalBiasAuditor()
    results = auditor.evaluate(y_true=y_actual, y_pred=y_ai, sensitive_features=race)

    print("\n--- Disaggregated Performance by Demographic Group ---")
    print(results.by_group.round(3).to_string())

    print("\n--- Absolute Disparity Differences ---")
    print(results.difference().round(3).to_string())

if __name__ == "__main__":
    main()
