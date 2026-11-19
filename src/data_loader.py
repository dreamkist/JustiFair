"""Data loader utilities for legal decision support benchmarks."""

from typing import Tuple
import pandas as pd

COMPAS_DATA_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
)

def load_compas_data(
    races: Tuple[str, ...] = ("African-American", "Caucasian")
) -> pd.DataFrame:
    """Download, clean, and filter the COMPAS recidivism benchmark dataset.

    Filters records according to established research standards:
    - Screening arrest occurred within 30 days of assessment.
    - Recidivism flag is valid.
    - Charge degree excludes minor traffic offenses.
    - Score text is available.
    """
    raw_df = pd.read_csv(COMPAS_DATA_URL)
    cleaned = raw_df[
        (raw_df["days_b_screening_arrest"] <= 30)
        & (raw_df["days_b_screening_arrest"] >= -30)
        & (raw_df["is_recid"] != -1)
        & (raw_df["c_charge_degree"] != "O")
        & (raw_df["score_text"] != "N/A")
    ].copy()

    if races:
        cleaned = cleaned[cleaned["race"].isin(races)].copy()

    return cleaned
