"""
Prompt Templates - FINAL VERSION
"""

POLICY_REVIEW_SYSTEM_PROMPT = """You are a Google Ads Policy Expert AI Assistant.

CRITICAL RULES:
1. Base ALL decisions on provided policy context only
2. Always cite exact policy language
3. Use clear decisions: allowed/restricted/disallowed/unclear
4. If unclear, set decision="unclear" and escalation_required=true
5. Extract specific violating phrases

CONFIDENCE CALIBRATION:
- 0.80-0.95: Clear policy match with exact violation/compliance identified
- 0.65-0.80: Strong policy match with good evidence
- 0.45-0.65: Moderate match, some ambiguity in application
- 0.30-0.45: Weak policy coverage or significant ambiguity
- Below 0.30: Insufficient policy information (use "unclear" decision)

Set confidence based on:
- Quality of policy match (how directly it addresses the ad)
- Clarity of violation (explicit vs implicit)
- Completeness of retrieved policy text

OUTPUT: Valid JSON matching PolicyDecision schema."""

POLICY_REVIEW_USER_PROMPT = """Review this ad against Google Ads policies.

AD CONTENT:
{ad_text}

RELEVANT POLICIES:
{policy_context}

Provide JSON with:
- decision: "allowed"|"restricted"|"disallowed"|"unclear"
- confidence: 0.0-1.0 (calibrated per guidelines above)
- policy_section: hierarchy (e.g., "Healthcare > Weight Loss")
- citation_url: official policy URL
- justification: clear explanation
- policy_quote: exact policy text
- risk_factors: list of violating phrases
- escalation_required: true if needs human review"""

POLICY_QA_SYSTEM_PROMPT = """You are a Google Ads Policy Expert.

Answer questions using ONLY provided policy context.
Always cite sources and quote policies.
If policy doesn't address question, say so explicitly."""

POLICY_QA_USER_PROMPT = """Answer this policy question.

QUESTION:
{question}

RELEVANT POLICIES:
{policy_context}

Provide JSON answer with citations."""


def format_policy_context(retrieved_chunks: list) -> str:
    """Format policy chunks for LLM"""
    formatted_parts = []
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        hierarchy = " > ".join(chunk['metadata']['hierarchy'])
        url = chunk['metadata']['url']
        content = chunk['content']
        
        formatted = f"""
POLICY {i}:
Category: {hierarchy}
Source: {url}

Content:
{content}

---
"""
        formatted_parts.append(formatted)
    
    return "\n".join(formatted_parts)


def format_policy_review_prompt(ad_text: str, policy_chunks: list) -> dict:
    """Create prompt for ad review"""
    policy_context = format_policy_context(policy_chunks)
    
    return {
        "system": POLICY_REVIEW_SYSTEM_PROMPT,
        "user": POLICY_REVIEW_USER_PROMPT.format(
            ad_text=ad_text,
            policy_context=policy_context
        )
    }


def format_policy_qa_prompt(question: str, policy_chunks: list) -> dict:
    """Create prompt for Q&A"""
    policy_context = format_policy_context(policy_chunks)
    
    return {
        "system": POLICY_QA_SYSTEM_PROMPT,
        "user": POLICY_QA_USER_PROMPT.format(
            question=question,
            policy_context=policy_context
        )
    }