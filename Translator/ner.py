import spacy

nlp = spacy.load("en_core_web_sm")

VALID_LABELS = ["PERSON", "ORG", "PRODUCT", "GPE"]


def mask_entities(text):
    doc = nlp(text)
    entities = {}
    masked_text = text
    counter = 1

    for ent in doc.ents:
        if ent.label_ in VALID_LABELS:
            placeholder = f"<ENT{counter}>"
            entities[placeholder] = ent.text
            masked_text = masked_text.replace(ent.text, placeholder)
            counter += 1

    return masked_text, entities


def restore_entities(text, entities):
    for placeholder, value in entities.items():
        text = text.replace(placeholder, value)
    return text