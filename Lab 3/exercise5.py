import os
import sys
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def translate_sentence(sentence, target_language, formality):
    system_prompt = (
        "You are a professional multi-language translator.\n"
        "Translate the input sentence into the requested target language at the specified formality level.\n"
        "Formality guidelines:\n"
        "- Formal: Uses polite pronouns, respectful vocabulary, and professional grammar.\n"
        "- Informal: Uses casual pronouns, conversational vocabulary, and relaxed grammar.\n"
        "- Neutral: Uses standard, natural phrasing without heavy formal or informal markers.\n\n"
        "Rules:\n"
        "1. Preserve the exact original meaning.\n"
        "2. Return only the translated sentence.\n"
        "3. Do not include pronunciation guides, notes, explanations, or quotes."
    )

    user_prompt = (
        f"Sentence: {sentence}\n"
        f"Target Language: {target_language}\n"
        f"Formality Level: {formality}\n\n"
        "Translation:"
    )

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )

    return response.choices[0].message.content.strip().strip('"')

if __name__ == "__main__":
    sentence = "Could you please let me know when the meeting starts?"

    test_cases = [
        ("Spanish", "Formal"),
        ("Spanish", "Informal"),
        ("German", "Formal"),
        ("German", "Informal"),
        ("French", "Formal"),
        ("French", "Informal")
    ]

    print(f"Original Sentence: {sentence}\n")
    for language, formality in test_cases:
        translated = translate_sentence(sentence, language, formality)
        print(f"{language} ({formality}): {translated}")
