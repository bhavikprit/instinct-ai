"""
Reflex Online Active Learning & Self-Tuning Instinct Head.
Pure-Python gradient descent runtime updating System 1 weights online with zero external dependencies.
"""

from __future__ import annotations
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from reflex.embeddings import SemanticVectorEncoder, cosine_similarity
from reflex.feedback import FeedbackItem


def _sigmoid(z: float) -> float:
    clamped_z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-clamped_z))


def _softmax(logits: List[float]) -> List[float]:
    if not logits:
        return []
    max_l = max(logits)
    exps = [math.exp(max(-30.0, min(30.0, x - max_l))) for x in logits]
    sum_exps = sum(exps)
    if sum_exps <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [e / sum_exps for e in exps]


class SelfTuningInstinctHead:
    """
    Online-trainable linear decision head sitting on top of 384-d semantic embeddings.
    Allows Reflex to learn from System 2 escalations and tune weights in <0.05ms per turn.
    """

    def __init__(
        self,
        dim: int = 384,
        encoder: Optional[SemanticVectorEncoder] = None,
    ):
        self.encoder = encoder or SemanticVectorEncoder()
        self.dim = getattr(self.encoder, "DIM", dim)
        # Weights per classification head: head_key -> List[float]
        self.weights: Dict[str, List[float]] = {}
        self.biases: Dict[str, float] = {}
        self.training_steps: int = 0

    def _ensure_head(self, head_key: str):
        """Initializes weights for a head if not already present."""
        if head_key not in self.weights:
            # Small random Xavier-like initialization
            rng = random.Random(hash(head_key) & 0xFFFFFFFF)
            std = 1.0 / math.sqrt(self.dim)
            self.weights[head_key] = [rng.gauss(0.0, std) for _ in range(self.dim)]
            self.biases[head_key] = 0.0

    def predict_noul(self, state_or_vec: Union[str, List[float]], question_key: str = "_default") -> float:
        """Returns calibrated probability [0.0 - 1.0] for a binary Noul decision."""
        vec = state_or_vec if isinstance(state_or_vec, list) else self.encoder.encode(state_or_vec)
        self._ensure_head(question_key)

        w = self.weights[question_key]
        b = self.biases[question_key]

        # Dot product
        z = sum(wi * xi for wi, xi in zip(w, vec)) + b
        return _sigmoid(z)

    def predict_choice(
        self,
        state_or_vec: Union[str, List[float]],
        options: List[str],
        prefix: str = "choice",
    ) -> Tuple[str, Dict[str, float]]:
        """Returns winning option and probability distribution across candidate options."""
        if not options:
            return "", {}
        vec = state_or_vec if isinstance(state_or_vec, list) else self.encoder.encode(state_or_vec)

        logits = []
        for opt in options:
            head_key = f"{prefix}::{opt}"
            self._ensure_head(head_key)
            w = self.weights[head_key]
            b = self.biases[head_key]
            z = sum(wi * xi for wi, xi in zip(w, vec)) + b
            logits.append(z)

        probs = _softmax(logits)
        dist = {opt: round(p, 4) for opt, p in zip(options, probs)}

        # Winner is max probability
        winning_idx = max(range(len(probs)), key=lambda i: probs[i])
        return options[winning_idx], dist

    def update_binary(
        self,
        state_or_vec: Union[str, List[float]],
        target: float,
        question_key: str = "_default",
        lr: float = 0.05,
        l2_reg: float = 0.001,
    ) -> float:
        """
        Performs an instantaneous online gradient descent step for a binary Noul.
        Returns the cross-entropy loss for the sample.
        """
        vec = state_or_vec if isinstance(state_or_vec, list) else self.encoder.encode(state_or_vec)
        self._ensure_head(question_key)

        w = self.weights[question_key]
        b = self.biases[question_key]

        z = sum(wi * xi for wi, xi in zip(w, vec)) + b
        p = _sigmoid(z)

        # Gradient of binary cross-entropy w.r.t logit is (p - y)
        target_val = 1.0 if target > 0.5 else 0.0
        error = p - target_val

        # Update weights and bias
        for i in range(self.dim):
            grad_w = error * vec[i] + l2_reg * w[i]
            w[i] -= lr * grad_w

        self.biases[question_key] = b - lr * error
        self.training_steps += 1

        # Binary cross-entropy loss
        eps = 1e-12
        loss = -(target_val * math.log(max(eps, p)) + (1.0 - target_val) * math.log(max(eps, 1.0 - p)))
        return round(loss, 5)

    def update_choice(
        self,
        state_or_vec: Union[str, List[float]],
        target_option: str,
        options: List[str],
        prefix: str = "choice",
        lr: float = 0.05,
        l2_reg: float = 0.001,
    ) -> float:
        """
        Performs an online gradient step for multi-class Choice routing.
        """
        vec = state_or_vec if isinstance(state_or_vec, list) else self.encoder.encode(state_or_vec)
        if target_option not in options:
            options = options + [target_option]

        logits = []
        for opt in options:
            head_key = f"{prefix}::{opt}"
            self._ensure_head(head_key)
            w = self.weights[head_key]
            b = self.biases[head_key]
            z = sum(wi * xi for wi, xi in zip(w, vec)) + b
            logits.append(z)

        probs = _softmax(logits)

        # Softmax cross-entropy gradients
        target_idx = options.index(target_option)
        for idx, opt in enumerate(options):
            head_key = f"{prefix}::{opt}"
            w = self.weights[head_key]
            b = self.biases[head_key]
            p = probs[idx]
            y = 1.0 if idx == target_idx else 0.0
            error = p - y

            for i in range(self.dim):
                grad_w = error * vec[i] + l2_reg * w[i]
                w[i] -= lr * grad_w
            self.biases[head_key] = b - lr * error

        self.training_steps += 1
        eps = 1e-12
        loss = -math.log(max(eps, probs[target_idx]))
        return round(loss, 5)

    def save_weights(self, filepath: str):
        """Serializes tuned weights to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "dim": self.dim,
            "training_steps": self.training_steps,
            "heads": list(self.weights.keys()),
            "weights": self.weights,
            "biases": self.biases,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_weights(self, filepath: str):
        """Loads serialized weights from JSON."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.dim = data["dim"]
        self.training_steps = data.get("training_steps", 0)
        self.weights = data["weights"]
        self.biases = data["biases"]


class OnlineTuner:
    """
    Offline and batch tuner for Reflex instinct models.
    """

    def __init__(
        self,
        head: Optional[SelfTuningInstinctHead] = None,
        lr: float = 0.05,
    ):
        self.head = head or SelfTuningInstinctHead()
        self.lr = lr

    def tune_on_samples(
        self,
        samples: List[FeedbackItem],
        epochs: int = 5,
    ) -> Dict[str, Any]:
        """Runs iterative gradient updates across collected feedback items."""
        if not samples:
            return {"epochs": 0, "samples": 0, "final_loss": 0.0}

        losses = []
        for ep in range(epochs):
            cur_lr = self.lr * (1.0 / (1.0 + 0.1 * ep))
            random.shuffle(samples)
            epoch_losses = []

            for item in samples:
                if isinstance(item.ground_truth, bool):
                    target_val = 1.0 if item.ground_truth else 0.0
                    loss = self.head.update_binary(
                        state_or_vec=item.state,
                        target=target_val,
                        question_key=item.question_key,
                        lr=cur_lr,
                    )
                    epoch_losses.append(loss)
                elif isinstance(item.ground_truth, str):
                    # Multi-class choice
                    options = [item.ground_truth]
                    if item.system1_prediction and item.system1_prediction != item.ground_truth:
                        options.append(item.system1_prediction)
                    loss = self.head.update_choice(
                        state_or_vec=item.state,
                        target_option=item.ground_truth,
                        options=options,
                        prefix=item.question_key,
                        lr=cur_lr,
                    )
                    epoch_losses.append(loss)

            mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            losses.append(mean_loss)

        return {
            "epochs": epochs,
            "samples": len(samples),
            "initial_loss": round(losses[0], 4) if losses else 0.0,
            "final_loss": round(losses[-1], 4) if losses else 0.0,
        }
