import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
from Config import MODEL_NAME


@st.cache_resource
def load_model():
    print("🔄 Loading Qwen model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    print("✅ Model loaded")
    return tokenizer, model


tokenizer, model = load_model()