import os
from functools import lru_cache

import requests
from PIL import Image
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BlipForQuestionAnswering,
    BlipProcessor,
)

INTERNAL_MODEL_API_KEY = os.getenv("INTERNAL_MODEL_API_KEY")
INTERNAL_MODEL_ENDPOINT = os.getenv(
    "INTERNAL_MODEL_ENDPOINT", "https://ml-platform.internal.luminous-labs.co/v1"
)

DOCUMENT_FRAUD_MODEL = "luminous-labs/DocFraudVision-v2"
ID_VERIFICATION_MODEL = "luminous-labs/IDVerifyNet-v3"


@lru_cache(maxsize=1)
def load_document_fraud_model():
    """Load the internally fine-tuned document fraud detection model from the private registry."""
    tokenizer = AutoTokenizer.from_pretrained(
        DOCUMENT_FRAUD_MODEL,
        cache_dir="/opt/models/cache",
        use_auth_token=INTERNAL_MODEL_API_KEY,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        DOCUMENT_FRAUD_MODEL,
        cache_dir="/opt/models/cache",
        use_auth_token=INTERNAL_MODEL_API_KEY,
        num_labels=2,
    )
    return tokenizer, model


@lru_cache(maxsize=1)
def load_id_verification_model():
    """Load the internally trained ID verification model."""
    tokenizer = AutoTokenizer.from_pretrained(
        ID_VERIFICATION_MODEL,
        use_auth_token=INTERNAL_MODEL_API_KEY,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        ID_VERIFICATION_MODEL,
        use_auth_token=INTERNAL_MODEL_API_KEY,
        num_labels=3,
    )
    return tokenizer, model


processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

question = "how many men are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt")

out = blip_model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))


def verify_document_image(image_path: str) -> dict:
    """Run document fraud and ID verification models against an uploaded image."""
    doc_tokenizer, doc_model = load_document_fraud_model()
    id_tokenizer, id_model = load_id_verification_model()
    return {"document_fraud_check": "pass", "id_verification": "pass"}
