import os
import sys
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_doctor_summary(patient_name, patient_notes):
    system_prompt = (
        "You are a medical assistant that converts patient consultation notes into a structured doctor summary.\n"
        "Rules:\n"
        "1. Always use this exact structure:\n"
        "Doctor Summary\n"
        "Patient: <patient_name>\n\n"
        "Symptoms:\n"
        "<list of symptoms or 'Not provided'>\n\n"
        "Diagnosis:\n"
        "<diagnosis or 'Not provided'>\n\n"
        "Recommendation:\n"
        "<recommendations or 'Not provided'>\n"
        "2. Use only information explicitly stated in the notes.\n"
        "3. Do not invent any symptoms, diagnosis, or recommendations.\n"
        "4. If any section is missing from the notes, write 'Not provided'.\n"
        "5. Do not include any introductory text, extra commentary, or concluding notes."
    )

    user_prompt = f"Patient Name: {patient_name}\nConsultation Notes: {patient_notes}"

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
    patient_name = "Rahul Sharma"
    patient_notes = (
        "Patient reports persistent sore throat, mild fever of 100.2 F, and dry cough for the past 3 days. "
        "Throat shows redness and swelling upon examination. Diagnosed with acute pharyngitis. "
        "Advised warm saline gargles three times daily, paracetamol 500mg as needed, and plenty of fluids."
    )

    summary = generate_doctor_summary(patient_name, patient_notes)
    print(summary)
