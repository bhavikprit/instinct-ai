"""
Reflex Example 3: Drop-in OpenAI-compatible Proxy Client.
Demonstrates sending an OpenAI-formatted request to the Reflex Proxy.
"""

import sys, os
import json
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_proxy_call():
    url = "http://127.0.0.1:8080/v1/chat/completions"
    
    # Standard OpenAI Chat Completion Request
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "Customer ticket: I was charged twice for renewal and need my money back immediately!"
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        print(f"Sending standard OpenAI request to Reflex Proxy ({url})...")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("\n✓ Intercepted & Resolved by Reflex System 1!")
            print(f"Model returned: {data.get('model')}")
            print(f"Latency: {data.get('reflex_meta', {}).get('latency_ms')} ms")
            print(f"Content: {data['choices'][0]['message']['content']}")
            print(f"Completion tokens billed: {data['usage']['completion_tokens']} (FREE)")
    except urllib.error.URLError:
        print("\nNote: Start the proxy first in another terminal via:")
        print("  python3 -m reflex.cli serve --port 8080")

if __name__ == "__main__":
    test_proxy_call()
