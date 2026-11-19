# JustiFair: Fairness and Bias Analysis in AI-Based Legal Decision Support

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Fairlearn](https://img.shields.io/badge/fairness-Fairlearn-green.svg)](https://fairlearn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**JustiFair** is an open-source auditing and mitigation toolkit designed to assess racial and demographic disparities in algorithmic legal decision support tools (such as pre-trial bail assessment, recidivism risk scoring, and sentencing recommendations).

---

## The Problem: Algorithmic Bias in the Justice System

When machine learning algorithms are introduced to assist judges in pre-trial detention and bail hearings, algorithmic disparities can lead to severe constitutional and human rights concerns:

* **Disparate False Positive Rates (FPR)**: Defendants misclassified as "high risk" are subjected to denied bail, harsher restrictions, or prolonged pre-trial incarceration.
* **Disparate False Negative Rates (FNR)**: Underestimating risk for specific demographic groups creates unequal safety assessments.
* **The COMPAS Case Study**: Audits of real-world commercial risk assessment tools demonstrated that minority defendants faced significantly higher False Positive Rates (wrongly predicted to recidivate) compared to other groups.

---

## Key Features

- **Disaggregated Legal Metrics**: Evaluate models on False Positive Rates, False Negative Rates, Selection Rates, and Positive Predictive Value across sensitive groups using `MetricFrame`.
- **Pre-Trial Risk Audit**: Built-in data loader for the canonical COMPAS recidivism benchmark.
- **In-Processing Mitigation**: Train fair models enforcing **Equalized Odds** without relying on sensitive attributes at test/decision time (mitigating Disparate Treatment legal risks).
- **Interactive Jupyter Tutorial**: Step-by-step walkthrough covering data preparation, baseline model audits, mitigation, and comparative visual analysis.

---

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/dreamkist/JustiFair.git
cd JustiFair

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart

### 1. Run the Command-Line Audit
Run an automated audit on the COMPAS benchmark dataset:

```bash
python scripts/run_audit.py
```

### 2. Interactive Notebook Walkthrough
Launch Jupyter to explore the step-by-step tutorial:

```bash
jupyter notebook notebooks/legal_bias_analysis_tutorial.ipynb
```

---

## Benchmark Audit Results

Evaluating the commercial COMPAS tool on actual two-year recidivism outcomes reveals the following disparities:

| Demographic Group | Count | High-Risk Flag Rate | False Positive Rate (*Wrongly Flagged High Risk*) | False Negative Rate (*Wrongly Flagged Low Risk*) |
| :--- | :---: | :---: | :---: | :---: |
| **African-American** | 3,175 | **58.8%** | **42.3%** | 28.5% |
| **Caucasian** | 2,103 | **34.3%** | **22.1%** | **49.6%** |
| **Disparity (Absolute Difference)** | — | **24.5%** | **20.3%** higher | **21.2%** higher |

Applying Fairlearn's `ExponentiatedGradient` with `EqualizedOdds` significantly closes this gap:

```python
from src.data_loader import load_compas_data
from src.mitigation import FairLegalClassifier
from src.audit import LegalBiasAuditor

# Load legal data
df = load_compas_data()
X = pd.get_dummies(df[['age', 'priors_count', 'c_charge_degree']], drop_first=True)
y = df['two_year_recid']
race = df['race']

# Train mitigated model under Equalized Odds constraint
classifier = FairLegalClassifier(constraint="equalized_odds")
classifier.fit(X, y, sensitive_features=race)

# Predict without needing race at inference time
y_pred = classifier.predict(X)

# Audit post-mitigation fairness
auditor = LegalBiasAuditor()
audit_results = auditor.evaluate(y_true=y, y_pred=y_pred, sensitive_features=race)
print(audit_results.difference())
```

---

## Key Legal & Sociotechnical Concepts

### 1. Impossibility Theorem of Fairness
Proven by Kleinberg et al. (2016) and Chouldechova (2017): When baseline outcome rates differ across demographic groups, an algorithm cannot simultaneously satisfy **Predictive Parity** (equal PPV) and **Equalized Odds** (equal FPR/FNR). Legal practitioners must make an explicit policy decision on which metric to prioritize.

### 2. Disparate Treatment vs. Disparate Impact
Under equal protection jurisprudence, models that require protected characteristics (like race or gender) as runtime inputs to adjust thresholds can face strict legal scrutiny. JustiFair uses in-processing reductions so that individual predictions are computed solely from legal case features.

---

## Project Structure

```
JustiFair/
├── notebooks/
│   └── legal_bias_analysis_tutorial.ipynb   # Interactive analysis & visualization
├── src/
│   ├── __init__.py
│   ├── data_loader.py                      # Data cleaning & loading utilities
│   ├── audit.py                            # MetricFrame legal audit wrapper
│   └── mitigation.py                       # Fairness-constrained classifiers
├── scripts/
│   └── run_audit.py                        # Command-line audit script
├── tests/
│   └── test_audit.py                       # Unit tests
├── requirements.txt                        # Dependency list
├── pyproject.toml                          # Package configuration
└── README.md                               # Project documentation
```

---

## License

This project is licensed under the MIT License.
