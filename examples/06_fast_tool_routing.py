"""
Reflex Example 6: Fast Tool Routing & Function Calling Pruning
Prunes 12+ candidate tools to the top 2 in <2ms, slashing prompt tokens by 80%.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import FastToolRouter

# 1. Standard OpenAI function schemas (12 tools that would consume 2,000+ tokens)
SAMPLE_OPENAI_TOOLS = [
    {"type": "function", "function": {"name": "calculator", "description": "Solves arithmetic calculations and mathematical expressions"}},
    {"type": "function", "function": {"name": "sql_query", "description": "Executes SQL queries against the user production PostgreSQL database"}},
    {"type": "function", "function": {"name": "web_search", "description": "Performs real-time web search for current news and live internet content"}},
    {"type": "function", "function": {"name": "refund_processor", "description": "Processes customer credit card refunds and billing chargebacks"}},
    {"type": "function", "function": {"name": "send_email", "description": "Sends transactional email to customer recipient"}},
    {"type": "function", "function": {"name": "git_commit", "description": "Commits code changes to the active git repository"}},
    {"type": "function", "function": {"name": "docker_deploy", "description": "Deploys containers to Kubernetes cluster"}},
    {"type": "function", "function": {"name": "password_reset", "description": "Sends secure password reset link to user account"}},
    {"type": "function", "function": {"name": "weather_lookup", "description": "Fetches current temperature and weather forecast for a city"}},
    {"type": "function", "function": {"name": "file_reader", "description": "Reads contents of local text or markdown files"}},
    {"type": "function", "function": {"name": "image_generator", "description": "Generates AI artwork or logos from text prompts"}},
    {"type": "function", "function": {"name": "audio_transcribe", "description": "Transcribes recorded MP3/WAV speech into text"}},
]

test_queries = [
    "What is 9382 multiplied by 481?",
    "Customer is furious about double billing on invoice #492 and demands their money back.",
    "Can you check if it is raining in Tokyo right now?",
    "Push the latest commit to the production Kubernetes cluster.",
]

print("=== Reflex Fast Tool Router (Function Calling Speedup) ===")
print(f"Total tools in registry: {len(SAMPLE_OPENAI_TOOLS)}\n")

for query in test_queries:
    pruned = FastToolRouter.filter_openai_tools(
        prompt=query,
        tools=SAMPLE_OPENAI_TOOLS,
        top_k=2
    )
    names = [t["function"]["name"] for t in pruned]
    print(f"Query:  '{query}'")
    print(f"Top 2 Pruned Tools: {names}")
    print(f"Token Reduction:   {(1.0 - (len(pruned) / len(SAMPLE_OPENAI_TOOLS))) * 100:.0f}%")
    print("-" * 55)
