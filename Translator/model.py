import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
from Config import MODEL_NAME


@st.cache_resource
def load_model():
    print("🔄 Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True
    )

    print("✅ Model loaded")
    return tokenizer, model


tokenizer, model = load_model()