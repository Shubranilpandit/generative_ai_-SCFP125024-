import os
import sys
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def classify_review(review_text):
    system_prompt = (
        "You are a customer review classifier.\n"
        "Classify the given review into exactly one of the following categories:\n"
        "- Positive: The customer expresses satisfaction, praise, or happiness with the product or service.\n"
        "- Negative: The customer expresses dissatisfaction, complaints, frustration, or problems.\n"
        "- Neutral: The customer expresses mixed feelings, ordinary expectations, or purely factual observations without clear sentiment.\n\n"
        "Rules:\n"
        "1. Return only the single category label: Positive, Negative, or Neutral.\n"
        "2. Do not include any punctuation, quotes, explanations, or additional words."
    )

    user_prompt = f"Customer Review:\n{review_text}\n\nCategory:"

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    sample_reviews = [
        "The battery life is incredible and the display is crystal clear. Best purchase this year!",
        "The package arrived broken and customer support was completely unhelpful.",
        "The headphones are okay. Sound quality is standard for this price point, nothing special."
    ]

    for review in sample_reviews:
        category = classify_review(review)
        print(f"Review: {review}")
        print(f"Category: {category}\n")
