import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from Config import MODEL_NAME

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("🔄 Loading model...")

        tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
        model = M2M100ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True
        )

        print("✅ Model loaded")
    return tokenizer, model