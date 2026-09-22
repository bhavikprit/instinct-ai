"""
Unit tests for Reflex Real-Time Concept Drift & OOD Guard (Phase 41 - reflex.drift).
Covers:
- DriftConfig validation & default hyperparameters
- Random Fourier Features (RFF) generation & determinism
- Fitting baseline distributions on raw text and vectors
- Cosine, Mahalanobis, and Euclidean distance metrics
- Empirical percentile OOD threshold calibration
- In-distribution vs Out-of-Distribution query evaluation
- Circular sliding window streaming updates & queue rollover
- Population Stability Index (PSI) and severity classification
- Streaming MMD two-sample test statistic and p-value calculation
- Automated callback triggering on drift
- Binary serialization (.reflex-drift) with CRC32 integrity verification
- Reflex client integration, Step 12 evaluation, and automated System-2 escalation
"""

import math
import os
import random
import tempfile
import unittest

from reflex.drift import (
    DriftConfig,
    DriftResult,
    DriftGuard,
    DRIFT_MAGIC,
    DRIFT_FORMAT_VERSION,
)
from reflex.embeddings import SemanticVectorEncoder
from reflex.client import Reflex
from reflex.primitives import Noul, Choice


class TestDriftConfig(unittest.TestCase):
    """Tests for DriftConfig hyperparameter validation."""

    def test_default_config(self):
        cfg = DriftConfig()
        self.assertEqual(cfg.window_size, 100)
        self.assertEqual(cfg.psi_threshold, 0.20)
        self.assertEqual(cfg.mmd_p_value_threshold, 0.05)
        self.assertEqual(cfg.ood_percentile, 95.0)
        self.assertEqual(cfg.num_rff_features, 64)
        self.assertEqual(cfg.rff_gamma, 0.5)
        self.assertEqual(cfg.psi_bins, 10)
        self.assertEqual(cfg.ood_metric, "cosine")
        self.assertEqual(cfg.seed, 42)

    def test_invalid_window_size(self):
        with self.assertRaises(ValueError):
            DriftConfig(window_size=5)

    def test_invalid_psi_threshold(self):
        with self.assertRaises(ValueError):
            DriftConfig(psi_threshold=0.0)
        with self.assertRaises(ValueError):
            DriftConfig(psi_threshold=2.5)

    def test_invalid_mmd_p_value(self):
        with self.assertRaises(ValueError):
            DriftConfig(mmd_p_value_threshold=0.0)
        with self.assertRaises(ValueError):
            DriftConfig(mmd_p_value_threshold=1.0)

    def test_invalid_ood_percentile(self):
        with self.assertRaises(ValueError):
            DriftConfig(ood_percentile=40.0)
        with self.assertRaises(ValueError):
            DriftConfig(ood_percentile=100.0)

    def test_invalid_rff_features(self):
        with self.assertRaises(ValueError):
            DriftConfig(num_rff_features=7)  # odd
        with self.assertRaises(ValueError):
            DriftConfig(num_rff_features=4)  # < 8

    def test_invalid_rff_gamma(self):
        with self.assertRaises(ValueError):
            DriftConfig(rff_gamma=0.0)

    def test_invalid_psi_bins(self):
        with self.assertRaises(ValueError):
            DriftConfig(psi_bins=1)

    def test_invalid_ood_metric(self):
        with self.assertRaises(ValueError):
            DriftConfig(ood_metric="manhattan")


class TestDriftGuardFittingAndDistances(unittest.TestCase):
    """Tests for fitting reference distributions and distance metric computations."""

    def setUp(self):
        self.encoder = SemanticVectorEncoder()
        self.banking_prompts = [
            "Check my checking account balance",
            "Transfer money to my savings account",
            "What is my account routing number?",
            "I want to order replacement checks",
            "Show me my recent debit card transactions",
            "Can I set up automatic bill payments?",
            "What are the fees for an international wire transfer?",
            "How do I deposit a check using the app?",
            "Is there a daily ATM cash withdrawal limit?",
            "I forgot my online banking password and need a reset",
            "Please freeze my lost credit card immediately",
            "What documents are needed to open a joint checking account?",
            "How do I activate my new debit card pin?",
            "Show me the bank statement for last month",
            "Can I set up travel alerts before going abroad?",
            "What is the minimum balance to avoid monthly maintenance fee?",
            "How long does an ACH transfer take to clear?",
            "Apply for a new auto loan refinancing",
            "Update my home address on my bank profile",
            "Schedule an appointment with a mortgage advisor",
            "Report fraudulent charge on my debit card",
            "What is the current savings interest APR?",
        ]

    def test_insufficient_reference_samples_error(self):
        guard = DriftGuard(DriftConfig(min_reference_samples=20))
        with self.assertRaises(ValueError):
            guard.fit(self.banking_prompts[:10])

    def test_evaluate_without_fitting_raises(self):
        guard = DriftGuard()
        with self.assertRaises(RuntimeError):
            guard.evaluate("Check balance")

    def test_fit_with_text_prompts_cosine(self):
        guard = DriftGuard(DriftConfig(ood_metric="cosine", seed=123))
        guard.fit(self.banking_prompts)
        self.assertTrue(guard.is_fitted)
        self.assertEqual(guard.num_reference_samples, len(self.banking_prompts))
        self.assertEqual(guard.dim, 384)
        self.assertEqual(len(guard.reference_centroid), 384)
        self.assertEqual(len(guard.reference_std), 384)
        self.assertGreater(guard.ood_threshold, 0.0)
        self.assertEqual(len(guard.psi_bin_edges), len(guard.reference_psi_frequencies) + 1)
        self.assertEqual(len(guard.reference_rff_mean), guard.config.num_rff_features)

    def test_fit_with_mahalanobis_metric(self):
        guard = DriftGuard(DriftConfig(ood_metric="mahalanobis", ood_percentile=90.0))
        guard.fit(self.banking_prompts)
        self.assertTrue(guard.is_fitted)
        self.assertGreater(guard.ood_threshold, 0.0)

    def test_fit_with_euclidean_metric(self):
        guard = DriftGuard(DriftConfig(ood_metric="euclidean", ood_percentile=95.0))
        guard.fit(self.banking_prompts)
        self.assertTrue(guard.is_fitted)
        self.assertGreater(guard.ood_threshold, 0.0)

    def test_fit_with_precomputed_vectors(self):
        vecs = [self.encoder.encode(p) for p in self.banking_prompts]
        guard = DriftGuard()
        guard.fit(vecs)
        self.assertTrue(guard.is_fitted)
        self.assertEqual(guard.num_reference_samples, len(vecs))


class TestDriftGuardEvaluation(unittest.TestCase):
    """Tests for individual query scoring, OOD detection, and streaming window updates."""

    def setUp(self):
        self.banking_prompts = [
            "Check checking account balance",
            "Transfer money to savings account",
            "What is my routing number?",
            "Order replacement checks",
            "Show recent debit transactions",
            "Set up automatic bill payment",
            "Fees for wire transfer",
            "Deposit check using mobile app",
            "Daily ATM withdrawal limit",
            "Forgot banking password reset",
            "Freeze lost credit card",
            "Open joint checking account",
            "Activate debit card pin",
            "Show bank statement for last month",
            "Set up travel alerts",
            "Minimum balance for fee waiver",
            "ACH transfer clearing time",
            "Apply for auto loan refinancing",
            "Update home mailing address",
            "Schedule appointment with advisor",
            "Report fraudulent charge",
            "Savings account interest rate",
        ]
        self.cfg = DriftConfig(
            window_size=30,
            psi_threshold=0.20,
            mmd_p_value_threshold=0.05,
            ood_percentile=95.0,
            min_reference_samples=20,
            seed=42,
        )
        self.guard = DriftGuard(config=self.cfg).fit(self.banking_prompts)

    def test_in_distribution_query(self):
        query = "Can you show me my checking balance?"
        res = self.guard.evaluate(query)
        self.assertIsInstance(res, DriftResult)
        self.assertFalse(res.is_ood)
        self.assertFalse(res.should_escalate)
        self.assertLessEqual(res.ood_score, self.guard.ood_threshold)
        self.assertEqual(res.sample_count, 1)
        self.assertEqual(res.window_count, 1)

    def test_out_of_distribution_query(self):
        # Adversarial / completely out-of-distribution code injection query
        query = "<script>alert('malware');</script> SELECT * FROM admin_passwords;"
        res = self.guard.evaluate(query)
        self.assertIsInstance(res, DriftResult)
        self.assertTrue(res.is_ood)
        self.assertTrue(res.should_escalate)
        self.assertGreater(res.ood_score, self.guard.ood_threshold)

    def test_window_rollover(self):
        # Window size is 30. Evaluate 45 samples and check window count stays at 30
        for i in range(45):
            self.guard.evaluate(f"Query {i} about account balance and transactions")
        self.assertEqual(self.guard._total_samples, 45)
        self.assertEqual(len(self.guard._window_vectors), 30)
        self.assertEqual(len(self.guard._window_scores), 30)

    def test_reset_window(self):
        for i in range(15):
            self.guard.evaluate("Check balance")
        self.assertEqual(self.guard._total_samples, 15)
        self.guard.reset_window()
        self.assertEqual(self.guard._total_samples, 0)
        self.assertEqual(len(self.guard._window_vectors), 0)
        self.assertEqual(len(self.guard._window_scores), 0)
        self.assertTrue(self.guard.is_fitted)  # Reference distribution intact

    def test_result_to_dict(self):
        res = self.guard.evaluate("Check balance")
        d = res.to_dict()
        self.assertIn("is_ood", d)
        self.assertIn("ood_score", d)
        self.assertIn("ood_threshold", d)
        self.assertIn("ood_metric", d)
        self.assertIn("psi", d)
        self.assertIn("psi_severity", d)
        self.assertIn("mmd_statistic", d)
        self.assertIn("mmd_p_value", d)
        self.assertIn("has_drift", d)
        self.assertIn("sample_count", d)
        self.assertIn("window_count", d)
        self.assertIn("should_escalate", d)


class TestDriftStatisticalDetection(unittest.TestCase):
    """Tests for PSI, MMD two-sample tests, and concept drift detection."""

    def setUp(self):
        self.banking_prompts = [
            "Check checking balance",
            "Transfer money to savings",
            "Account routing number",
            "Order replacement checks",
            "Recent debit transactions",
            "Automatic bill pay setup",
            "Wire transfer fees",
            "Mobile check deposit",
            "Daily ATM withdrawal limit",
            "Forgot password reset",
            "Freeze credit card",
            "Open joint checking",
            "Activate debit card pin",
            "Bank statement last month",
            "Travel alert setup",
            "Minimum balance waiver",
            "ACH transfer time",
            "Apply auto loan refinancing",
            "Update mailing address",
            "Schedule advisor appointment",
        ]
        self.cfg = DriftConfig(
            window_size=40,
            psi_threshold=0.15,
            psi_moderate_threshold=0.08,
            mmd_p_value_threshold=0.05,
            ood_percentile=95.0,
            min_reference_samples=20,
            seed=42,
        )
        self.guard = DriftGuard(config=self.cfg).fit(self.banking_prompts)

    def test_in_distribution_stream_remains_stable(self):
        rng = random.Random(42)
        last_res = None
        for _ in range(35):
            last_res = self.guard.evaluate(rng.choice(self.banking_prompts))

        self.assertIsNotNone(last_res.psi)
        self.assertLess(last_res.psi, 0.15)
        self.assertEqual(last_res.psi_severity, "none")
        self.assertFalse(last_res.has_drift)

    def test_drifted_stream_triggers_psi_and_mmd(self):
        # Crypto / DeFi vocabulary shift
        crypto_samples = [
            "Swap Ethereum for Solana on Uniswap decentralized liquidity pool",
            "What is the Arbitrum gas fee for a token swap transaction?",
            "Stake Solana tokens in validator node for staking rewards",
            "Bridge Bitcoin across cross-chain bridge smart contract",
            "Execute perpetual futures trade on decentralized exchange",
            "Check memecoin transaction volume on Dexscreener",
            "Mint NFT on OpenSea with MetaMask hardware wallet",
            "Yield farming liquidity pool impermanent loss calculation",
        ]
        rng = random.Random(42)
        last_res = None
        for _ in range(35):
            last_res = self.guard.evaluate(rng.choice(crypto_samples))

        self.assertIsNotNone(last_res.psi)
        self.assertIsNotNone(last_res.mmd_p_value)
        # Should detect significant shift
        self.assertTrue(last_res.has_drift)
        self.assertIn(last_res.psi_severity, ["moderate", "severe"])
        self.assertLess(last_res.mmd_p_value, 0.05)

    def test_drift_detected_callback_fires(self):
        callback_fired = []

        def on_drift(res: DriftResult):
            callback_fired.append(res)

        guard = DriftGuard(
            config=self.cfg,
            on_drift_detected=on_drift,
        ).fit(self.banking_prompts)

        crypto_samples = [
            "Swap Ethereum for Solana on Uniswap decentralized liquidity pool",
            "Arbitrum gas fee for token swap transaction",
            "Stake Solana tokens in validator node",
            "Bridge Bitcoin cross-chain smart contract",
        ]
        rng = random.Random(42)
        for _ in range(35):
            guard.evaluate(rng.choice(crypto_samples))

        self.assertGreater(len(callback_fired), 0)
        self.assertTrue(callback_fired[-1].has_drift)

    def test_ascii_drift_report(self):
        for _ in range(25):
            self.guard.evaluate("Check account balance and transfer funds")
        report = self.guard.ascii_drift_report()
        self.assertIn("Reflex Real-Time Concept Drift & OOD Report", report)
        self.assertIn("Reference Samples", report)
        self.assertIn("PSI Quantile Distribution", report)


class TestDriftGuardBinaryPersistence(unittest.TestCase):
    """Tests for saving and loading .reflex-drift binary models with CRC32 verification."""

    def setUp(self):
        self.prompts = [
            f"Banking customer inquiry sample prompt {i} with account details"
            for i in range(25)
        ]
        self.cfg = DriftConfig(
            window_size=50,
            psi_threshold=0.18,
            ood_percentile=92.0,
            ood_metric="mahalanobis",
            num_rff_features=32,
            min_reference_samples=20,
            seed=99,
        )
        self.guard = DriftGuard(config=self.cfg).fit(self.prompts)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.reflex-drift")
            self.guard.save(path)
            self.assertTrue(os.path.exists(path))

            loaded = DriftGuard.load(path)
            self.assertTrue(loaded.is_fitted)
            self.assertEqual(loaded.dim, self.guard.dim)
            self.assertEqual(loaded.num_reference_samples, self.guard.num_reference_samples)
            self.assertAlmostEqual(loaded.ood_threshold, self.guard.ood_threshold, places=6)
            self.assertEqual(loaded.config.ood_metric, "mahalanobis")
            self.assertEqual(loaded.config.num_rff_features, 32)
            self.assertEqual(loaded.config.psi_threshold, 0.18)

            # Test evaluation parity between original and loaded
            test_query = "What is my account balance?"
            res_orig = self.guard.evaluate(test_query)
            res_loaded = loaded.evaluate(test_query)
            self.assertEqual(res_orig.is_ood, res_loaded.is_ood)
            self.assertAlmostEqual(res_orig.ood_score, res_loaded.ood_score, places=5)

    def test_load_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            DriftGuard.load("/nonexistent/path/model.reflex-drift")

    def test_load_corrupt_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrupt.reflex-drift")
            with open(path, "wb") as f:
                f.write(b"CORRUPT DATA TOO SHORT")
            with self.assertRaises(ValueError):
                DriftGuard.load(path)

    def test_crc32_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tampered.reflex-drift")
            self.guard.save(path)
            # Tamper with one byte in body
            with open(path, "rb") as f:
                data = bytearray(f.read())
            data[60] = (data[60] + 1) % 256
            with open(path, "wb") as f:
                f.write(data)
            with self.assertRaises(ValueError) as ctx:
                DriftGuard.load(path)
            self.assertIn("CRC32", str(ctx.exception))


class TestReflexClientDriftIntegration(unittest.TestCase):
    """Tests for Reflex client runtime integration with DriftGuard (Step 12)."""

    def setUp(self):
        self.prompts = [
            "Check checking balance",
            "Transfer money to savings account",
            "What is my routing number?",
            "Order replacement checks",
            "Show recent debit transactions",
            "Set up automatic bill payment",
            "Fees for wire transfer",
            "Deposit check using mobile app",
            "Daily ATM withdrawal limit",
            "Forgot banking password reset",
            "Freeze lost credit card",
            "Open joint checking account",
            "Activate debit card pin",
            "Show bank statement for last month",
            "Set up travel alerts",
            "Minimum balance for fee waiver",
            "ACH transfer clearing time",
            "Apply for auto loan refinancing",
            "Update home mailing address",
            "Schedule appointment with advisor",
            "Report fraudulent charge",
            "Savings account interest rate",
        ]
        self.guard = DriftGuard(
            config=DriftConfig(min_reference_samples=20, ood_percentile=95.0)
        ).fit(self.prompts)

    def test_client_in_distribution_no_escalation(self):
        rx = Reflex(drift_guard=self.guard)
        res = rx.evaluate(
            "What is my current checking account balance?",
            {"is_valid": Noul(instructions="Is valid query")},
        )
        self.assertIsNotNone(res.drift)
        self.assertFalse(res.drift.is_ood)
        # Should not force escalate
        self.assertFalse(res.should_escalate)
        self.assertIn("drift", res.to_dict())

    def test_client_ood_triggers_automatic_escalation(self):
        rx = Reflex(drift_guard=self.guard)
        # Attack / OOD payload
        ood_input = "<script>alert('xss');</script> DROP DATABASE credentials;"
        res = rx.evaluate(
            ood_input,
            {"is_valid": Noul(instructions="Is valid query")},
        )
        self.assertIsNotNone(res.drift)
        self.assertTrue(res.drift.is_ood)
        # Automatically flags System-2 escalation
        self.assertTrue(res.should_escalate)

    def test_client_record_drift_sample(self):
        rx = Reflex(drift_guard=self.guard)
        res = rx.record_drift_sample("Check checking balance and transfer money")
        self.assertIsNotNone(res)
        self.assertIsInstance(res, DriftResult)
        self.assertFalse(res.is_ood)


if __name__ == "__main__":
    unittest.main()
