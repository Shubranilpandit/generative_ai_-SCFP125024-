#Exercise 1 - Doctor Summary Generator
#Build a prompt that takes a patient's raw consultation notes and generates a clean doctor's summary.
#Input : patient_notes, patient_name
#Output : a summary with the fixed sections -- Symptoms, Diagnosis, Recommendations -- in the same format every time, regardless of input


from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ["GROQ_API_KEY"])


# System prompt: defines the AI's role and output format
SYSTEM_PROMPT = """
You are a Doctor Summary Generator.

Your task is to convert raw patient consultation notes into a
clean, concise, and structured doctor's summary.

Follow these rules strictly:
1. Use the patient's name provided in the input.
2. Use only information available in the consultation notes.
3. Never invent symptoms, diagnoses, medicines, or recommendations.
4. Do not create a new diagnosis. Report only what the notes document.
5. If information is missing, write "Not documented in the notes."
6. Correct spelling and grammar without changing the medical meaning.
7. Always use the exact output structure given below.
8. Do not add any extra sections or introductory text.

Required output format:

Doctor's Summary
Patient Name: <patient_name>

1. Symptoms
- <symptoms identified from the notes>

2. Diagnosis
- <diagnosis documented in the notes, or state that it is not documented>

3. Recommendations
- <recommendations documented in the notes, or state that they are not documented>
"""


# Function to generate the summary
def generate_doctor_summary(patient_notes, patient_name):

    user_prompt = f"""
    Patient Name: {patient_name}

    Raw Consultation Notes:
    {patient_notes}

    Generate the doctor's summary using the required format.
    """

    chat_completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    return chat_completion.choices[0].message.content


# Example input
patient_name = "Rahul Sharma"

patient_notes = """
Patient has been experiencing fever, headache, and body aches
for the last two days. The doctor suspects a viral infection.
The patient is advised to take adequate rest, drink fluids,
and monitor body temperature.
"""

# Generate and display the summary
summary = generate_doctor_summary(patient_notes, patient_name)

print(summary)
