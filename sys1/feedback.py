"""
Reflex Feedback & Escalation Collector.
Captures System 2 reasoning ground truth and uncertainty escalations to power active learning.
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class FeedbackItem:
    """Represents a single ground-truth training sample collected from System 2."""
    state: str
    question_key: str
    ground_truth: Union[bool, str, float]
    system1_prediction: Optional[Any] = None
    source_model: str = "claude-3.5-sonnet"
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "question_key": self.question_key,
            "ground_truth": self.ground_truth,
            "system1_prediction": self.system1_prediction,
            "source_model": self.source_model,
            "timestamp": self.timestamp,
        }


class FeedbackCollector:
    """
    In-memory and file-backed buffer for collecting agent escalation feedback.
    Provides training pairs for Reflex's self-tuning instinct memory.
    """

    def __init__(
        self,
        max_buffer_size: int = 2000,
        log_file: Optional[str] = None,
    ):
        self.max_buffer_size = max_buffer_size
        self.log_file = log_file
        self.items: List[FeedbackItem] = []

    def record(
        self,
        state: str,
        question_key: str,
        ground_truth: Union[bool, str, float],
        system1_prediction: Optional[Any] = None,
        source_model: str = "claude-3.5-sonnet",
    ) -> FeedbackItem:
        """Records a new feedback observation."""
        item = FeedbackItem(
            state=state,
            question_key=question_key,
            ground_truth=ground_truth,
            system1_prediction=system1_prediction,
            source_model=source_model,
        )

        if len(self.items) >= self.max_buffer_size:
            self.items.pop(0)

        self.items.append(item)

        if self.log_file:
            self._append_to_file(item)

        return item

    def _append_to_file(self, item: FeedbackItem):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(item.to_dict()) + "\n")
        except Exception:
            pass

    def get_samples(self, question_key: Optional[str] = None) -> List[FeedbackItem]:
        """Returns collected items, optionally filtered by question_key."""
        if question_key is None:
            return list(self.items)
        return [item for item in self.items if item.question_key == question_key]

    def export_jsonl(self, filepath: str):
        """Exports all in-memory items to a JSONL dataset file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps(item.to_dict()) + "\n")

    def load_jsonl(self, filepath: str):
        """Loads items from an existing JSONL dataset."""
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.record(
                    state=d["state"],
                    question_key=d["question_key"],
                    ground_truth=d["ground_truth"],
                    system1_prediction=d.get("system1_prediction"),
                    source_model=d.get("source_model", "external"),
                )

    def clear(self):
        self.items.clear()

    @property
    def count(self) -> int:
        return len(self.items)
