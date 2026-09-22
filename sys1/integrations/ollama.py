"""
Reflex Ollama Local Dual-Brain Bridge.
Pairs Reflex (System 1) with local Ollama models (System 2) for 100% offline agent loops.
"""

from __future__ import annotations
import asyncio
import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from sys1.client import Reflex
from sys1.primitives import Noul, Choice


class OllamaDualBrain:
    """
    Offline Dual-Brain Orchestrator.
    
    Routes incoming prompts through Reflex (System 1) in <1ms.
    Only spins up heavy local Ollama weights when epistemic uncertainty is high,
    saving MacBook / GPU thermal throttling and battery.
    
    Usage:
        brain = OllamaDualBrain(model="llama3.2")
        
        # High confidence check (instant <1ms, Ollama never touched!)
        res = brain.chat(
            prompt="URGENT: Click here to claim your $1000 prize immediately!",
            noul_question="Is this a phishing scam?"
        )
        print(res["resolved_by"]) # -> "reflex"
        print(res["ollama_called"]) # -> False
    """

    def __init__(
        self,
        model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434",
        epistemic_threshold: float = 0.85,
        reflex_client: Optional[Reflex] = None,
    ):
        self.model = model
        self.base_url = ollama_base_url.rstrip("/")
        self.epistemic_threshold = epistemic_threshold
        self.rx = reflex_client or Reflex(backend="local")

    def chat(
        self,
        prompt: str,
        noul_question: Optional[str] = None,
        choice_options: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes a dual-brain chat turn.
        Reflex filters and routes at the spinal cord; Ollama awakens only on doubt.
        """
        start_time = time.perf_counter()

        # Step 1: System 1 Reflex Pass
        if noul_question:
            prob = self.rx.noul(noul_question, prompt)
            is_confident = (prob >= self.epistemic_threshold) or (prob <= (1.0 - self.epistemic_threshold))
            if is_confident:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "content": "true" if prob >= self.epistemic_threshold else "false",
                    "probability": prob,
                    "resolved_by": "reflex",
                    "ollama_called": False,
                    "latency_ms": round(elapsed_ms, 2),
                    "cost_usd": 0.0,
                }

        if choice_options:
            choice_val = self.rx.choice("Select route", choice_options, prompt)
            # Evaluate distribution confidence
            res = self.rx.evaluate(prompt, {"_c": Choice("Select", options=choice_options)})
            top_prob = res["_c"].distribution.get(choice_val, 0.0)
            if top_prob >= self.epistemic_threshold:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "content": choice_val,
                    "probability": top_prob,
                    "resolved_by": "reflex",
                    "ollama_called": False,
                    "latency_ms": round(elapsed_ms, 2),
                    "cost_usd": 0.0,
                }

        # Step 2: System 2 Escalation (Ollama)
        ollama_response = self._call_ollama_generate(prompt, system_prompt=system_prompt)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "content": ollama_response,
            "resolved_by": "ollama",
            "ollama_called": True,
            "model": self.model,
            "latency_ms": round(elapsed_ms, 2),
            "cost_usd": 0.0,  # 100% offline and free
        }

    async def achat(
        self,
        prompt: str,
        noul_question: Optional[str] = None,
        choice_options: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Asynchronously executes dual-brain turn."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.chat,
            prompt,
            noul_question,
            choice_options,
            system_prompt,
        )

    def _call_ollama_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Dispatches request to local Ollama API endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {url}. Is Ollama running? (ollama serve). Error: {e}"
            ) from e
