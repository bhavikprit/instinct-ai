"""
AsyncReflex: Asynchronous Non-Blocking System 1 Runtime Client.
Provides native asyncio bindings for high-throughput production agent loops.
"""

from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from sys1.client import Reflex
from sys1.primitives import PrimitiveType, DecisionResult


class AsyncReflex:
    """
    AsyncReflex: Native asynchronous interface for Reflex System 1 decisions.
    
    Usage:
        import asyncio
        from sys1 import AsyncReflex, Noul, Choice
        
        async def main():
            rx = AsyncReflex()
            
            # Non-blocking evaluation
            result = await rx.aevaluate(
                state="Customer requesting refund for duplicate invoice",
                questions={
                    "is_refund": Noul("Does the user demand a refund?"),
                    "queue": Choice("Route queue", ["billing", "support", "sales"])
                }
            )
            
            # Inline non-blocking shortcuts
            prob = await rx.anoul("Is this phishing?", email_body)
            action = await rx.achoice("Next action", ["search", "calc", "finish"], context)
            
        asyncio.run(main())
    """

    def __init__(
        self,
        backend: Union[str, Any] = "auto",
        policy: str = "dual-brain",
        api_key: Optional[str] = None,
        max_workers: int = 8,
        **kwargs,
    ):
        self._sync_client = Reflex(
            backend=backend,
            policy=policy,
            api_key=api_key,
            **kwargs,
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def aevaluate(
        self,
        state: str,
        questions: Dict[str, PrimitiveType],
    ) -> DecisionResult:
        """Asynchronously evaluates state against questions in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_client.evaluate,
            state,
            questions,
        )

    async def anoul(
        self,
        instructions: str,
        state: str,
        threshold: float = 0.85,
    ) -> float:
        """Asynchronously returns calibrated probability float [0.0 - 1.0]."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_client.noul,
            instructions,
            state,
            threshold,
        )

    async def achoice(
        self,
        instructions: str,
        options: List[str],
        state: str,
    ) -> str:
        """Asynchronously returns winning option string."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_client.choice,
            instructions,
            options,
            state,
        )

    async def ascore(
        self,
        instructions: str,
        state: str,
        min_val: float = 1.0,
        max_val: float = 10.0,
    ) -> float:
        """Asynchronously returns continuous evaluated score."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_client.score,
            instructions,
            state,
            min_val,
            max_val,
        )

    def close(self):
        """Clean up thread executor resources."""
        self._executor.shutdown(wait=False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
