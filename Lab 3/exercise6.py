import os
import sys
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def refine_content(draft_text, issue_to_fix, total_rounds=3):
    system_prompt = (
        "You are an iterative content refiner.\n"
        "Your goal is to revise the provided draft text specifically to address the indicated issue.\n"
        "Rules:\n"
        "1. Focus solely on resolving the specified issue.\n"
        "2. Do not unnecessarily alter unrelated parts or change the underlying meaning.\n"
        "3. Return only the revised text with no introductory phrases, commentary, or quotes."
    )

    current_text = draft_text
    rounds_history = []

    for round_number in range(1, total_rounds + 1):
        user_prompt = (
            f"Current Draft:\n{current_text}\n\n"
            f"Specific Issue to Fix:\n{issue_to_fix}\n\n"
            f"Round {round_number} Revision:"
        )

        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        current_text = response.choices[0].message.content.strip().strip('"')
        rounds_history.append((round_number, current_text))

    return rounds_history, current_text

if __name__ == "__main__":
    draft_text = "Hey team, we gotta hurry up and finish this project ASAP cuz the client is getting super angry and might cancel stuff."
    issue_to_fix = "Make the tone professional, calm, and suitable for a formal corporate update email."

    print(f"Original Draft:\n{draft_text}\n")
    print(f"Issue to Fix:\n{issue_to_fix}\n")
    print("=" * 60)

    rounds, final_version = refine_content(draft_text, issue_to_fix, total_rounds=3)

    for round_num, revised_text in rounds:
        print(f"Round {round_num} Output:\n{revised_text}\n")
        print("-" * 60)

    print(f"Final Refined Version:\n{final_version}")
