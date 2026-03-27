import os
from functools import lru_cache

from smolagents.agents import ZeroShotAgent
from smolagents.llms import OpenAIWrapper
from smolagents.memory import Memory
from smolagents.tools import DuckDuckGoSearchTool, PythonREPLTool
from transformers import AutoModelForCausalLM, AutoTokenizer

INTERNAL_MODEL_API_KEY = os.getenv("INTERNAL_MODEL_API_KEY")
INTERNAL_MODEL_ENDPOINT = os.getenv(
    "INTERNAL_MODEL_ENDPOINT", "https://ml-platform.internal.luminous-labs.co/v1"
)

FRAUD_REASONING_MODEL = "luminous-labs/FraudReasoningLLM-v2"
TRANSACTION_ANOMALY_MODEL = "luminous-labs/TxnAnomalyDetector-v1"


@lru_cache(maxsize=1)
def load_fraud_reasoning_model():
    """Load the internally fine-tuned fraud reasoning LLM from the private registry."""
    tokenizer = AutoTokenizer.from_pretrained(
        FRAUD_REASONING_MODEL,
        cache_dir="/opt/models/cache",
        use_auth_token=INTERNAL_MODEL_API_KEY,
    )
    model = AutoModelForCausalLM.from_pretrained(
        FRAUD_REASONING_MODEL,
        cache_dir="/opt/models/cache",
        use_auth_token=INTERNAL_MODEL_API_KEY,
    )
    return tokenizer, model


@lru_cache(maxsize=1)
def load_transaction_anomaly_model():
    """Load the internally trained transaction anomaly detection model."""
    tokenizer = AutoTokenizer.from_pretrained(
        TRANSACTION_ANOMALY_MODEL,
        use_auth_token=INTERNAL_MODEL_API_KEY,
    )
    model = AutoModelForCausalLM.from_pretrained(
        TRANSACTION_ANOMALY_MODEL,
        use_auth_token=INTERNAL_MODEL_API_KEY,
    )
    return tokenizer, model


memory = Memory()

llm = OpenAIWrapper(api_key=os.environ.get("OPENAI_KEY"), model_name="gpt-4")

repl_tool = PythonREPLTool()
search_tool = DuckDuckGoSearchTool()
tools = [repl_tool, search_tool]

agent = ZeroShotAgent(
    llm=llm,
    memory=memory,
    tools=tools,
    verbose=True,
)

task = "Calculate the square root of 144 and then multiply the result by 5."


def run_with_fraud_check(task: str) -> str:
    """Run agent task with fraud reasoning model loaded for supplemental analysis."""
    fraud_tokenizer, fraud_model = load_fraud_reasoning_model()
    anomaly_tokenizer, anomaly_model = load_transaction_anomaly_model()
    return agent.run(task)


if __name__ == "__main__":
    result = run_with_fraud_check(task)
    print(f"\nResult: {result}")
