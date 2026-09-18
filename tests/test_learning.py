"""
Unit tests for Reflex FeedbackCollector, SelfTuningInstinctHead, and online active learning.
"""

import os
import tempfile
import unittest

from reflex.feedback import FeedbackCollector, FeedbackItem
from reflex.learning import SelfTuningInstinctHead, OnlineTuner
from reflex.client import Reflex
from reflex.primitives import Noul, Choice


class TestFeedbackAndLearning(unittest.TestCase):

    def test_feedback_collector(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            collector = FeedbackCollector(max_buffer_size=3, log_file=tmp_path)
            collector.record("query 1", "is_urgent", True)
            collector.record("query 2", "is_urgent", False)
            collector.record("query 3", "department", "billing")
            collector.record("query 4", "department", "tech")

            # Buffer size clamped to 3
            self.assertEqual(collector.count, 3)

            # File has all 4 appended records
            with open(tmp_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 4)

            # Filtering by key
            dept_samples = collector.get_samples(question_key="department")
            self.assertEqual(len(dept_samples), 2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_binary_online_update(self):
        head = SelfTuningInstinctHead()
        state = "CRITICAL: Database primary node failover timed out"
        key = "is_p0"

        # Initially around 0.5
        p_init = head.predict_noul(state, question_key=key)

        # Train multiple times with target=1.0
        for _ in range(15):
            head.update_binary(state, target=1.0, question_key=key, lr=0.15)

        p_trained = head.predict_noul(state, question_key=key)
        self.assertGreater(p_trained, p_init)
        self.assertGreater(p_trained, 0.70)

    def test_choice_online_update(self):
        head = SelfTuningInstinctHead()
        state = "I need an urgent invoice receipt for my tax return"
        options = ["billing", "tech_support", "sales"]

        # Train that this query maps to 'billing'
        for _ in range(8):
            head.update_choice(state, target_option="billing", options=options, prefix="route", lr=0.1)

        selected, dist = head.predict_choice(state, options, prefix="route")
        self.assertEqual(selected, "billing")
        self.assertGreater(dist["billing"], dist["tech_support"])

    def test_weights_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            head = SelfTuningInstinctHead()
            head.update_binary("System alert test", target=1.0, question_key="test_head")
            head.save_weights(tmp_path)

            loaded_head = SelfTuningInstinctHead()
            loaded_head.load_weights(tmp_path)

            p1 = head.predict_noul("System alert test", question_key="test_head")
            p2 = loaded_head.predict_noul("System alert test", question_key="test_head")
            self.assertAlmostEqual(p1, p2, places=4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_online_tuner_loss_reduction(self):
        head = SelfTuningInstinctHead()
        tuner = OnlineTuner(head=head, lr=0.1)

        samples = [
            FeedbackItem("Database crash error 500", "is_bug", True),
            FeedbackItem("Server kernel panic", "is_bug", True),
            FeedbackItem("How to make iced latte at home", "is_bug", False),
            FeedbackItem("Weather in Paris today", "is_bug", False),
        ]

        stats = tuner.tune_on_samples(samples, epochs=10)
        self.assertEqual(stats["samples"], 4)
        self.assertLess(stats["final_loss"], stats["initial_loss"])

    def test_client_teach_integration(self):
        rx = Reflex(backend="local", learning=True)
        query = "Custom proprietary acronym: XYZ-99 alert"
        
        # Teach Reflex that this is a critical incident
        loss1 = rx.teach(query, "is_critical", ground_truth=True, lr=0.15)
        for _ in range(5):
            rx.teach(query, "is_critical", ground_truth=True, lr=0.15)

        res = rx.evaluate(query, {"is_critical": Noul("Is this critical?")})
        self.assertGreater(res["is_critical"].probability, 0.65)


if __name__ == "__main__":
    unittest.main()
