from Translator.model import tokenizer, model
from Translator.ner import mask_entities, restore_entities
from Config import LANGUAGES
from concurrent.futures import ThreadPoolExecutor, as_completed


def translate_text(text, target_lang):
    prompt = f"""
You are a professional translator.

Translate the following text into {target_lang}.

Rules:
- Do NOT translate names, brands, or placeholders like <ENT1>
- Keep tone natural and fluent
- Preserve meaning exactly

Text:
{text}
"""

    messages = [
        {"role": "system", "content": "You are a translation assistant."},
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.1,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in result:
        result = result.split("assistant")[-1]

    return result.strip()


def translate_pipeline(text, lang_code):
    target_lang = LANGUAGES[lang_code]

    masked_text, entities = mask_entities(text)
    translated = translate_text(masked_text, target_lang)
    final_text = restore_entities(translated, entities)

    return final_text


def translate_all(text):
    results = {}

    def task(lang):
        return lang, translate_pipeline(text, lang)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(task, code) for code in LANGUAGES]

        for future in as_completed(futures):
            lang, result = future.result()
            results[lang] = result

    return results