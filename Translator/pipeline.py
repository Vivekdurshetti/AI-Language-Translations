from Translator.model import tokenizer, model
from Translator.ner import mask_entities, restore_entities
from Config import LANGUAGES
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch


def translate_text(text, target_lang):
    prompt = f"Translate to {target_lang}: {text}"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the translation part (remove the prompt)
    if ":" in result:
        result = result.split(":", 1)[1].strip()

    return result


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