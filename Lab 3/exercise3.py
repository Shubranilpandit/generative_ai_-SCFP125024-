import os
import sys
import json
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_resume_fields(resume_text, fields_to_extract):
    system_prompt = (
        "You are a resume field extractor.\n"
        "Extract only the requested fields from the given resume text.\n"
        "Rules:\n"
        "1. Output must be a valid, strict JSON object only.\n"
        "2. Do not wrap the output in markdown code blocks like ```json or ```.\n"
        "3. The JSON object must contain only the keys listed in the requested fields.\n"
        "4. If a requested field is not mentioned or unavailable in the resume, set its value to null.\n"
        "5. Do not invent, infer, or assume any missing details.\n"
        "6. Do not include any explanations, greetings, or text outside the JSON object."
    )

    user_prompt = (
        f"Resume Text:\n{resume_text}\n\n"
        f"Fields to Extract:\n{json.dumps(fields_to_extract)}\n\n"
        "Return strict JSON:"
    )

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )

    raw_output = response.choices[0].message.content.strip()
    clean_json = raw_output.replace("```json", "").replace("```", "").strip()
    return clean_json

if __name__ == "__main__":
    resume_text = (
        "Priya Verma\n"
        "Email: priya.verma@example.com\n"
        "Phone: +91 9876543210\n"
        "Education: Master of Computer Applications, XYZ University (2024)\n"
        "Technical Skills: Python, SQL, Machine Learning, FastAPI, Git\n"
        "Work Experience: Software Engineering Intern at TechCorp for 6 months, building REST APIs."
    )

    fields_to_extract = ["name", "email", "education", "certifications", "skills"]
    extracted_json = extract_resume_fields(resume_text, fields_to_extract)
    print(extracted_json)
