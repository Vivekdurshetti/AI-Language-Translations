from Translator.pipeline import translate_all
from Config import LANGUAGES


def main():
    print("🌍 AI Translator (Fast Qwen)\n")

    text = input("Enter text: ").strip()

    if not text:
        print("No input provided")
        return

    print("\nTranslating...\n")

    results = translate_all(text)

    print("===== RESULTS =====")
    for lang, output in results.items():
        print(f"\n[{lang.upper()} - {LANGUAGES[lang]}]")
        print(output)


if __name__ == "__main__":
    main()