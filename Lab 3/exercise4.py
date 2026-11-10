import os
import sys
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_support_reply(customer_message, company_name, max_words):
    system_prompt = (
        f"You are a professional customer support representative for {company_name}.\n"
        "Rules:\n"
        "1. Write a polite, empathetic, and helpful reply addressing the customer's issue.\n"
        f"2. Your response must NEVER exceed {max_words} words.\n"
        "3. Maintain a consistent, reassuring, and solution-oriented tone.\n"
        "4. Output only the customer reply without any greetings to the user, word counts, or explanations."
    )

    user_prompt = (
        f"Company: {company_name}\n"
        f"Word Limit: {max_words} words\n"
        f"Customer Message: {customer_message}\n\n"
        "Support Reply:"
    )

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    reply = response.choices[0].message.content.strip()
    words = reply.split()
    if len(words) > max_words:
        reply = " ".join(words[:max_words])

    return reply

if __name__ == "__main__":
    company_name = "QuickShop"
    max_words = 35
    customer_message = "I ordered a wireless mouse 10 days ago (Order #78901) and it has still not been delivered. Can you please check what happened?"

    reply = generate_support_reply(customer_message, company_name, max_words)
    print(f"Customer Message: {customer_message}\n")
    print(f"Support Reply: {reply}\n")
    print(f"Word Count: {len(reply.split())} / {max_words}")
