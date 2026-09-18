"""
Reflex OpenRLCD: Reinforcement Learning from Calibrated Decisions.
Dataset curation and loss functions for System 1 decision post-training.
"""

from reflex.rlcd.loss import (
    brier_score,
    expected_calibration_error,
    epistemic_entropy,
    reliability_diagram_data,
)
from reflex.rlcd.dataset import (
    generate_decision_dataset,
    save_dataset_jsonl,
)

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "epistemic_entropy",
    "reliability_diagram_data",
    "generate_decision_dataset",
    "save_dataset_jsonl",
]
