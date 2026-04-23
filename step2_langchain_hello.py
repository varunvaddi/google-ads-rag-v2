"""
Step 2 — First LangChain call
Concepts: ChatOllama, Messages, invoke()
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ── 1. Create the LLM ──────────────────────────────────────────────────────
# This connects to Ollama running locally on port 11434
# No API key, no rate limits, runs forever for free
llm = ChatOllama(
    model="llama3.2",
    temperature=0.1,   # Low = more consistent (good for policy decisions)
)

# ── 2. Messages ────────────────────────────────────────────────────────────
# Instead of passing a raw string, LangChain uses Message objects
# SystemMessage = sets the role/behavior of the LLM
# HumanMessage  = the actual query/input
messages = [
    SystemMessage(content="You are a Google Ads policy expert. Be concise."),
    HumanMessage(content="In one sentence, can I advertise cryptocurrency on Google Ads?"),
]

# ── 3. Invoke ──────────────────────────────────────────────────────────────
# .invoke() sends the messages and waits for the full response
# (later we'll use .stream() for streaming — but invoke() first)
print("Calling llama3.2 via LangChain...")
response = llm.invoke(messages)

# ── 4. Read the response ───────────────────────────────────────────────────
# response is an AIMessage object — not a plain string
# .content is the actual text you care about
print(f"\nType:    {type(response)}")
print(f"\nAnswer:  {response.content}")