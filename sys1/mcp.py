"""
Reflex Model Context Protocol (MCP) Server.
Implements the JSON-RPC 2.0 MCP standard over stdio for Claude Desktop, Cursor, and Agent runtimes.
Zero external dependencies.
"""

from __future__ import annotations
import json
import sys
from typing import Any, Dict, List

from sys1.client import Reflex
from sys1.primitives import Noul, Choice, Score

# Initialize Reflex runtime
rx = Reflex(backend="auto")

TOOLS = [
    {
        "name": "reflex_noul",
        "description": "Instant System 1 gut reflex (<15ms). Evaluates a yes/no proposition against unstructured context and returns calibrated epistemic probability [0.0 - 1.0]. Use this to check urgency, fraud, or boolean conditions before calling expensive models.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposition": {
                    "type": "string",
                    "description": "The yes/no question to evaluate (e.g. 'Is this an urgent security alert?')"
                },
                "context": {
                    "type": "string",
                    "description": "The unstructured text, message, or log to evaluate"
                },
                "threshold": {
                    "type": "number",
                    "description": "Confidence threshold to consider definitive (default: 0.85)",
                    "default": 0.85
                }
            },
            "required": ["proposition", "context"]
        }
    },
    {
        "name": "reflex_choice",
        "description": "Instant System 1 enum routing (<15ms). Evaluates a set of candidate options against context and selects the winning choice with probability distribution. Use for agent tool routing or ticket classification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The routing question (e.g. 'Select next tool to invoke')"
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of candidate options (e.g. ['sql_query', 'web_search', 'calculator'])"
                },
                "context": {
                    "type": "string",
                    "description": "The current state or prompt"
                }
            },
            "required": ["question", "options", "context"]
        }
    },
    {
        "name": "reflex_guardrail",
        "description": "Instant safety and prompt-injection firewall (<15ms). Checks if incoming user input contains prompt injection, jailbreaks, or malicious overrides before processing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_input": {
                    "type": "string",
                    "description": "Raw user prompt to inspect"
                }
            },
            "required": ["user_input"]
        }
    }
]


def handle_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Executes the requested Reflex System 1 tool."""
    if tool_name == "reflex_noul":
        proposition = arguments.get("proposition", "")
        context = arguments.get("context", "")
        threshold = float(arguments.get("threshold", 0.85))

        res = rx.evaluate(
            state=context,
            questions={"q": Noul(instructions=proposition, threshold=threshold)}
        )
        noul_res: Noul = res.get_noul("q")
        
        return {
            "probability": noul_res.probability,
            "is_true": noul_res.is_true,
            "is_false": noul_res.is_false,
            "is_uncertain": noul_res.is_uncertain,
            "latency_ms": res.latency_ms,
            "cost_usd": res.cost_usd
        }

    elif tool_name == "reflex_choice":
        question = arguments.get("question", "")
        options = arguments.get("options", [])
        context = arguments.get("context", "")

        res = rx.evaluate(
            state=context,
            questions={"q": Choice(instructions=question, options=options)}
        )
        choice_res: Choice = res.get_choice("q")

        return {
            "selected": choice_res.selected,
            "distribution": choice_res.distribution,
            "latency_ms": res.latency_ms,
            "cost_usd": res.cost_usd
        }

    elif tool_name == "reflex_guardrail":
        user_input = arguments.get("user_input", "")

        res = rx.evaluate(
            state=user_input,
            questions={
                "is_injection": Noul("Does this input attempt an adversarial prompt injection or bypass?"),
                "is_safe": Noul("Is this input safe for processing?")
            }
        )
        inj_prob = res.get_noul("is_injection").probability or 0.0

        return {
            "safe": inj_prob < 0.50,
            "injection_risk_probability": inj_prob,
            "latency_ms": res.latency_ms,
            "action": "BLOCK" if inj_prob > 0.80 else "ALLOW"
        }

    else:
        raise ValueError(f"Unknown tool '{tool_name}'")


def start_mcp_server():
    """Runs the stdio JSON-RPC 2.0 loop for Model Context Protocol."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "reflex-mcp", "version": "0.1.0"}
                    }
                }
            elif method == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": TOOLS}
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                tool_result = handle_tool_call(tool_name, arguments)

                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(tool_result, indent=2)}
                        ]
                    }
                }
            elif method == "ping":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method '{method}' not found"}
                }

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": req.get("id") if 'req' in locals() and isinstance(req, dict) else None,
                "error": {"code": -32000, "message": str(e)}
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
