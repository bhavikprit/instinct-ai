"""
Unit tests for AsyncReflex asynchronous runtime.
"""

import asyncio
import unittest
from reflex.async_client import AsyncReflex
from reflex.primitives import Noul, Choice, Score


class TestAsyncReflex(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.rx = AsyncReflex(backend="local")

    async def asyncTearDown(self):
        self.rx.close()

    async def test_aevaluate(self):
        res = await self.rx.aevaluate(
            state="Critical database downtime emergency",
            questions={
                "urgent": Noul("Is this urgent?"),
                "action": Choice("Action", options=["reboot", "ignore"]),
            }
        )
        self.assertIn("urgent", res.decisions)
        self.assertIn("action", res.decisions)
        self.assertGreater(res["urgent"].probability, 0.6)

    async def test_anoul_and_achoice_shortcuts(self):
        prob = await self.rx.anoul("Is this a refund?", "I demand my money back for invoice #444")
        self.assertGreater(prob, 0.70)

        selected = await self.rx.achoice("Pick team", ["billing", "sales", "security"], "Password hacked!")
        self.assertEqual(selected, "security")

    async def test_ascore_shortcut(self):
        score = await self.rx.ascore("Rate urgency 1-10", "Routine non-urgent weekly newsletter", min_val=1.0, max_val=10.0)
        self.assertTrue(1.0 <= score <= 10.0)

    async def test_concurrent_evaluations(self):
        # Test 10 parallel queries via asyncio.gather
        tasks = [
            self.rx.anoul("Is this scam?", f"Urgent wire request #{i}")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        self.assertEqual(len(results), 10)
        for r in results:
            self.assertTrue(0.0 <= r <= 1.0)

    async def test_async_context_manager(self):
        async with AsyncReflex(backend="local") as rx:
            p = await rx.anoul("Is this urgent?", "Fire in the server room!")
            self.assertGreater(p, 0.70)


if __name__ == "__main__":
    unittest.main()
