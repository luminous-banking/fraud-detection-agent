import http.client
import json
import os
import ssl
from functools import lru_cache

from transformers import AutoModelForSequenceClassification, AutoTokenizer

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_HOST = "api.openai.com"
OPENAI_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
OPENAI_MODEL = "gpt-3.5-turbo"

INTERNAL_MODEL_API_KEY = os.getenv("INTERNAL_MODEL_API_KEY")
INTERNAL_MODEL_ENDPOINT = os.getenv(
    "INTERNAL_MODEL_ENDPOINT", "https://ml-platform.internal.luminous-labs.co/v1"
)

INTENT_CLASSIFIER_MODEL = "luminous-labs/FraudIntentClassifier-v4"
PROMPT_INJECTION_MODEL = "luminous-labs/PromptGuard-v1"


@lru_cache(maxsize=1)
def load_intent_classifier():
    """Load the internally fine-tuned fraud intent classifier from the private registry."""
    tokenizer = AutoTokenizer.from_pretrained(
        INTENT_CLASSIFIER_MODEL,
        cache_dir="/opt/models/cache",
        use_auth_token=INTERNAL_MODEL_API_KEY,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        INTENT_CLASSIFIER_MODEL,
        cache_dir="/opt/models/cache",
        use_auth_token=INTERNAL_MODEL_API_KEY,
        num_labels=5,
    )
    return tokenizer, model


@lru_cache(maxsize=1)
def load_prompt_guard():
    """Load the internally trained prompt injection detection model."""
    tokenizer = AutoTokenizer.from_pretrained(
        PROMPT_INJECTION_MODEL,
        use_auth_token=INTERNAL_MODEL_API_KEY,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        PROMPT_INJECTION_MODEL,
        use_auth_token=INTERNAL_MODEL_API_KEY,
        num_labels=2,
    )
    return tokenizer, model


def classify_intent(message: str) -> dict:
    """Run the intent classifier on an incoming message before sending to the LLM."""
    tokenizer, model = load_intent_classifier()
    inputs = tokenizer(message, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    return {"logits": outputs.logits.tolist()}


def check_prompt_safety(message: str) -> bool:
    """Screen user input for prompt injection attempts using the internal guard model."""
    tokenizer, model = load_prompt_guard()
    inputs = tokenizer(message, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predicted = outputs.logits.argmax(dim=-1).item()
    return predicted == 0


def get_chat_response(prompt_message):
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("Error: Please replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key.")
        return "API key not set."

    if not check_prompt_safety(prompt_message):
        return "Request blocked by safety filter."

    try:
        context = ssl._create_unverified_context()
        conn = http.client.HTTPSConnection(OPENAI_API_HOST, context=context)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }

        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt_message}],
            "temperature": 0.7,
            "max_tokens": 150,
        }

        json_payload = json.dumps(payload)
        conn.request("POST", OPENAI_CHAT_COMPLETIONS_PATH, body=json_payload, headers=headers)

        response = conn.getresponse()
        response_data = response.read().decode("utf-8")
        conn.close()

        if response.status != 200:
            print(f"Error: HTTP Status {response.status} - {response.reason}")
            print(f"Response Body: {response_data}")
            return f"API Error: {response.status} - {response.reason}"

        response_json = json.loads(response_data)

        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            return "No response from chatbot."

    except http.client.HTTPException as e:
        print(f"HTTP Error: {e}")
        return f"Network Error: {e}"
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return f"Data Error: {e}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}"


def main():
    print("Welcome to the simple OpenAI Chatbot!")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        intent = classify_intent(user_input)
        print(f"[Intent: {intent}]")

        print("Chatbot: Thinking...")
        chatbot_response = get_chat_response(user_input)
        print(f"Chatbot: {chatbot_response}")


if __name__ == "__main__":
    main()
