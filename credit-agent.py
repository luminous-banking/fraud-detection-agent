import os

from agents import Agent, AgentTypes

MISTRAL_KEY = os.getenv("MISTRAL_KEY")

credit_agent = Agent(
    agent_type=AgentTypes.TEXT.value,
    provider="mistral",
    mission="Analyze credit transactions and detect potential fraud patterns",
    model_params={"key": MISTRAL_KEY, "model": "mistral-medium"},
)