from Translator.model import load_model
from Translator.ner import mask_entities, restore_entities
from Config import LANGUAGES
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch


def translate_text(text, target_lang_code):
    tokenizer, model = load_model()
    tokenizer.src_lang = "en"
    tokenizer.tgt_lang = target_lang_code

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang_code),
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return result


def translate_pipeline(text, lang_code):
    masked_text, entities = mask_entities(text)
    translated = translate_text(masked_text, lang_code)
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