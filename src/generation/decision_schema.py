"""
Policy Decision Schema - FINAL VERSION
Structured output format using Pydantic
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from datetime import datetime


class PolicyDecision(BaseModel):
    """Structured decision for ad policy review"""
    
    decision: Literal["allowed", "restricted", "disallowed", "unclear"] = Field(
        description="Policy decision for the ad"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in decision (0.0-1.0)"
    )
    
    policy_section: str = Field(
        description="Hierarchical policy section"
    )
    
    citation_url: str = Field(
        description="URL to official Google Ads policy"
    )
    
    justification: str = Field(
        description="Explanation of decision"
    )
    
    policy_quote: str = Field(
        description="Exact policy quote"
    )
    
    risk_factors: Optional[List[str]] = Field(
        default=None,
        description="Specific violating phrases"
    )
    
    escalation_required: bool = Field(
        default=False,
        description="Needs human review"
    )
    
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="When decision was made"
    )


class PolicyQuestion(BaseModel):
    """Response for policy Q&A"""
    
    answer: str = Field(description="Answer to question")
    policy_section: str = Field(description="Relevant policy")
    citation_url: str = Field(description="Policy URL")
    policy_quote: str = Field(description="Policy excerpt")
    confidence: float = Field(ge=0.0, le=1.0)
    follow_up_needed: bool = Field(default=False)