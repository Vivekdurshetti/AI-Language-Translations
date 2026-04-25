import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
from Config import MODEL_NAME


@st.cache_resource
def load_model():
    print("🔄 Loading Qwen model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load model on CPU for deployment environments
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map={"": "cpu"}  # Explicitly map to CPU
    )

    print("✅ Model loaded")
    return tokenizer, model


tokenizer, model = load_model()