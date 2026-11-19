"""JustiFair: Legal AI Fairness & Bias Auditing Toolkit."""

from .data_loader import load_compas_data
from .audit import LegalBiasAuditor
from .mitigation import FairLegalClassifier

__version__ = "0.1.0"
__all__ = ["load_compas_data", "LegalBiasAuditor", "FairLegalClassifier"]
