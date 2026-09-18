"""
Unit tests for Reflex Web Playground HTTP Handler.
Zero-network in-memory testing for sandbox and CI/CD compatibility.
"""

import io
import json
import unittest
from unittest.mock import MagicMock

from reflex.web.playground import PlaygroundHandler


class MockSocket:
    """Mock socket for SimpleHTTPRequestHandler without opening network connections."""
    def __init__(self, request_bytes: bytes):
        self.rfile = io.BytesIO(request_bytes)
        self.wfile = io.BytesIO()
        self.sent_data = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        if "b" in mode:
            if "r" in mode:
                return self.rfile
            elif "w" in mode:
                return self.wfile
        return self.rfile

    def sendall(self, data):
        self.sent_data.write(data)

    def close(self):
        pass


class TestPlaygroundHandler(unittest.TestCase):

    def _execute_request(self, method: str, path: str, body: bytes = b"") -> tuple[int, dict, bytes]:
        """Runs an HTTP request through PlaygroundHandler in memory."""
        request_line = f"{method} {path} HTTP/1.1\r\n"
        headers = f"Host: localhost\r\nContent-Length: {len(body)}\r\n\r\n"
        full_request = request_line.encode("utf-8") + headers.encode("utf-8") + body

        mock_socket = MockSocket(full_request)
        mock_server = MagicMock()

        handler = PlaygroundHandler(mock_socket, ("127.0.0.1", 8000), mock_server)
        response_bytes = mock_socket.wfile.getvalue() or mock_socket.sent_data.getvalue()

        # Parse response status
        lines = response_bytes.split(b"\r\n")
        status_line = lines[0].decode("utf-8")
        status_code = int(status_line.split(" ")[1])

        # Separate headers and body
        header_body_split = response_bytes.split(b"\r\n\r\n", 1)
        resp_body = header_body_split[1] if len(header_body_split) > 1 else b""

        return status_code, {}, resp_body

    def test_get_root_html(self):
        status, _, body = self._execute_request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Reflex Dual-Brain Playground", body)

    def test_get_health(self):
        status, _, body = self._execute_request("GET", "/health")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["status"], "ok")

    def test_post_evaluate_high_confidence(self):
        payload = {
            "state": "CRITICAL EMERGENCY: Database breached by unknown IP",
            "noul_question": "Is this a critical security emergency?",
            "choice_question": "Security action",
            "choice_options": ["quarantine", "allow"],
            "gate_threshold": 0.80,
        }
        body = json.dumps(payload).encode("utf-8")
        status, _, resp_body = self._execute_request("POST", "/api/evaluate", body=body)

        self.assertEqual(status, 200)
        data = json.loads(resp_body.decode("utf-8"))
        self.assertIn("sys1", data)
        self.assertIn("sys2", data)
        self.assertIn("escalated", data)
        self.assertIn("noul_probability", data["sys1"])
        self.assertIn("choice_selected", data["sys1"])

    def test_post_evaluate_escalation(self):
        # Ambiguous query designed to fall into the epistemic gap
        payload = {
            "state": "I'm thinking about canceling maybe next month if budget cuts hit us.",
            "noul_question": "Is customer definitively canceling today?",
            "gate_threshold": 0.90,
        }
        body = json.dumps(payload).encode("utf-8")
        status, _, resp_body = self._execute_request("POST", "/api/evaluate", body=body)

        self.assertEqual(status, 200)
        data = json.loads(resp_body.decode("utf-8"))
        self.assertTrue(data["escalated"])
        self.assertTrue(data["sys2"]["awakened"])
        self.assertGreater(data["sys2"]["latency_ms"], 0)


if __name__ == "__main__":
    unittest.main()
