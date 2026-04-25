import streamlit as st
from Translator.pipeline import translate_pipeline
from Config import LANGUAGES

st.set_page_config(page_title="AI Translator", layout="wide")

st.title("🌍 AI Translation Tool (Fast Qwen)")

text = st.text_area("Enter text:")

selected_langs = st.multiselect(
    "Select languages",
    list(LANGUAGES.keys()),
    default=["de", "es", "fr"]
)


@st.cache_data
def cached_translate(text, langs):
    results = {}
    for lang in langs:
        results[lang] = translate_pipeline(text, lang)
    return results


if st.button("Translate"):
    if not text.strip():
        st.warning("Enter text first")
    else:
        with st.spinner("Translating... ⚡"):
            results = cached_translate(text, tuple(selected_langs))

        for lang in selected_langs:
            st.subheader(f"{lang.upper()} - {LANGUAGES[lang]}")
            st.write(results[lang])