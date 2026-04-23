"""
Step 3 — Structured output with Pydantic
Concepts: with_structured_output(), getting PolicyDecision directly from LLM
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Literal, Optional, List


# ── 1. Define a simple Pydantic schema ────────────────────────────────────
# This is a simplified version of your PolicyDecision
# LangChain will force the LLM to return data matching this shape exactly
class SimpleDecision(BaseModel):
    decision: Literal["allowed", "restricted", "disallowed", "unclear"] = Field(
        description="Policy decision for the ad"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0"
    )
    justification: str = Field(
        description="One sentence explanation of the decision"
    )
    risk_factors: Optional[List[str]] = Field(
        default=None,
        description="Specific phrases in the ad that violate policy"
    )


# ── 2. Create LLM and bind the schema ─────────────────────────────────────
# with_structured_output() tells LangChain:
# "force the LLM to respond in this exact shape"
# Internally it adds JSON instructions to the prompt automatically
llm = ChatOllama(model="llama3.2", temperature=0.1)
structured_llm = llm.with_structured_output(SimpleDecision)


# ── 3. Test with two ads ───────────────────────────────────────────────────
test_ads = [
    "Lose 15 pounds in one week with this miracle pill! Guaranteed results!",
    "Buy our new laptop - Intel i7, 16GB RAM, free shipping over $50",
]

for ad in test_ads:
    print(f"\n{'='*60}")
    print(f"Ad: {ad}")
    print(f"{'='*60}")

    messages = [
        SystemMessage(content="""You are a Google Ads policy expert.
Review the ad and return a structured decision.
Base your decision on common Google Ads policies:
- Health claims must be accurate and not misleading
- Miracle/guaranteed results are prohibited
- Standard product ads are generally allowed"""),

        HumanMessage(content=f"Review this ad for policy compliance:\n\n{ad}"),
    ]

    # ── 4. Invoke — response is already a SimpleDecision object ───────────
    # No json.loads(), no manual parsing, no try/except for JSON errors
    decision = structured_llm.invoke(messages)

    # decision IS a SimpleDecision Pydantic object — access fields directly
    print(f"Type:          {type(decision)}")
    print(f"Decision:      {decision.decision.upper()}")
    print(f"Confidence:    {decision.confidence:.1%}")
    print(f"Justification: {decision.justification}")
    print(f"Risk factors:  {decision.risk_factors}")

# ── 5. Fix: add a validator to clamp confidence to 0.0-1.0 ───────────────
# LLMs sometimes ignore numeric range instructions (gave confidence as 80 instead of 0.80)
# Pydantic validators catch this automatically

from pydantic import field_validator

class SimpleDecisionV2(BaseModel):
    decision: Literal["allowed", "restricted", "disallowed", "unclear"] = Field(
        description="Policy decision for the ad"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0"
    )
    justification: str = Field(
        description="One sentence explanation of the decision"
    )
    risk_factors: Optional[List[str]] = Field(
        default=None,
        description="Specific phrases in the ad that violate policy"
    )

    # Validator: if LLM returns 80 instead of 0.80, fix it automatically
    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v):
        if v > 1.0:
            v = v / 100.0   # convert 80 → 0.80
        return round(max(0.0, min(1.0, v)), 2)  # clamp to 0-1


# Rebind with fixed schema
structured_llm_v2 = llm.with_structured_output(SimpleDecisionV2)

print("\n\n--- Running with validator fix ---")

for ad in test_ads:
    print(f"\n{'='*60}")
    print(f"Ad: {ad}")
    print(f"{'='*60}")

    messages = [
        SystemMessage(content="""You are a Google Ads policy expert.
Review the ad and return a structured decision.
- Miracle/guaranteed health results are prohibited (disallowed)
- Standard product ads are generally fine (allowed)
- Crypto/alcohol/political ads need special approval (restricted)
- confidence MUST be between 0.0 and 1.0 (e.g. 0.85 not 85)"""),

        HumanMessage(content=f"Review this ad:\n\n{ad}"),
    ]

    decision = structured_llm_v2.invoke(messages)

    print(f"Decision:      {decision.decision.upper()}")
    print(f"Confidence:    {decision.confidence:.1%}")
    print(f"Justification: {decision.justification}")
    print(f"Risk factors:  {decision.risk_factors}")