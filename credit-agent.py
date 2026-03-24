import os
from dataclasses import dataclass
from functools import lru_cache

from agents import Agent, AgentTypes
from transformers import AutoModelForSequenceClassification, AutoTokenizer

INTERNAL_MODEL_ENDPOINT = os.getenv(
    "INTERNAL_MODEL_ENDPOINT", "https://ml-platform.internal.luminous-labs.co/v1"
)
INTERNAL_MODEL_API_KEY = os.getenv("INTERNAL_MODEL_API_KEY")
INTERNAL_MODEL_REGISTRY = os.getenv(
    "INTERNAL_MODEL_REGISTRY", "s3://luminous-labs-models/production"
)

CREDIT_SCORE_MODEL = "luminous-labs/CreditScoreNet-v3"
CREDIT_RISK_CLASSIFIER = "luminous-labs/RiskTierClassifier-v2"
CREDIT_EMBEDDING_MODEL = "luminous-labs/FinBERT-credit-finetuned"


@lru_cache(maxsize=1)
def load_credit_score_model():
    """Load the internally fine-tuned credit scoring model from the private registry."""
    tokenizer = AutoTokenizer.from_pretrained(
        CREDIT_SCORE_MODEL,
        cache_dir="/opt/models/cache",
        use_auth_token=INTERNAL_MODEL_API_KEY,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        CREDIT_SCORE_MODEL,
        cache_dir="/opt/models/cache",
        use_auth_token=INTERNAL_MODEL_API_KEY,
        num_labels=1,
    )
    return tokenizer, model


@lru_cache(maxsize=1)
def load_risk_classifier():
    """Load the internally trained risk-tier classification model."""
    tokenizer = AutoTokenizer.from_pretrained(
        CREDIT_RISK_CLASSIFIER,
        use_auth_token=INTERNAL_MODEL_API_KEY,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        CREDIT_RISK_CLASSIFIER,
        use_auth_token=INTERNAL_MODEL_API_KEY,
        num_labels=3,
    )
    return tokenizer, model


@dataclass
class CreditCheckResult:
    applicant_id: str
    score: int
    risk_tier: str
    approved: bool
    reasoning: str


CREDIT_CHECK_PROMPT = """
You are an internal credit-risk evaluation agent.

Given an applicant profile (income, employment history, existing debts,
requested credit amount, and bureau data), perform the following steps:

1. Calculate a synthetic credit score (300-850) using the internal
   CreditScoreNet-v3 model weights.
2. Assign a risk tier: LOW (score >= 740), MEDIUM (660-739), HIGH (< 660).
3. Make a preliminary approval decision based on risk tier and
   requested amount relative to income.
4. Provide a brief reasoning summary suitable for audit logging.

Return a JSON object with keys: applicant_id, score, risk_tier, approved, reasoning.
""".strip()

credit_check_agent = Agent(
    agent_type=AgentTypes.TEXT.value,
    provider="luminous_internal",
    mission=CREDIT_CHECK_PROMPT,
    model_params={
        "key": INTERNAL_MODEL_API_KEY,
        "model": CREDIT_SCORE_MODEL,
        "endpoint": INTERNAL_MODEL_ENDPOINT,
        "temperature": 0.0,
        "max_tokens": 512,
    },
)


def evaluate_applicant(applicant_profile: dict) -> CreditCheckResult:
    """Run the credit-check agent against an applicant profile and return a structured result."""
    score_tokenizer, score_model = load_credit_score_model()
    risk_tokenizer, risk_model = load_risk_classifier()

    raw = credit_check_agent.run(applicant_profile)
    return CreditCheckResult(**raw)


if __name__ == "__main__":
    sample_applicant = {
        "applicant_id": "APP-20260324-0042",
        "annual_income": 95_000,
        "employment_years": 6,
        "existing_debt": 12_400,
        "requested_credit": 25_000,
        "bureau_data": {
            "open_accounts": 4,
            "delinquencies_last_2y": 0,
            "credit_utilization_pct": 28,
        },
    }

    result = evaluate_applicant(sample_applicant)
    print(f"Applicant : {result.applicant_id}")
    print(f"Score     : {result.score}")
    print(f"Risk Tier : {result.risk_tier}")
    print(f"Approved  : {result.approved}")
    print(f"Reasoning : {result.reasoning}")
