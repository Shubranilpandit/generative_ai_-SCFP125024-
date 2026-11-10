import os
import sys
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

class BMIInput(BaseModel):
    weight_kg: float = Field(gt=0, description="Weight of the individual in kilograms")
    height_cm: float = Field(gt=0, description="Height of the individual in centimeters")

class DailyCalorieInput(BaseModel):
    weight_kg: float = Field(gt=0, description="Weight of the individual in kilograms")
    height_cm: float = Field(gt=0, description="Height of the individual in centimeters")
    age: int = Field(gt=0, description="Age of the individual in full years")
    activity_level: str = Field(description="Activity level such as sedentary, light, moderate, active, or very_active")

def compute_bmi(weight_kg: float, height_cm: float) -> str:
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    bmi_rounded = round(bmi, 2)
    if bmi_rounded < 18.5:
        category = "Underweight"
    elif bmi_rounded < 25.0:
        category = "Normal weight"
    elif bmi_rounded < 30.0:
        category = "Overweight"
    else:
        category = "Obesity"
    return json.dumps({
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "bmi": bmi_rounded,
        "category": category
    })

def compute_daily_calories(weight_kg: float, height_cm: float, age: int, activity_level: str) -> str:
    bmr = (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age) + 5.0
    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }
    selected_level = "moderate"
    for key in multipliers:
        if key in activity_level.lower():
            selected_level = key
            break
    multiplier = multipliers[selected_level]
    daily_calories = bmr * multiplier
    return json.dumps({
        "basal_metabolic_rate_bmr": round(bmr, 2),
        "activity_level": selected_level,
        "daily_maintenance_calories": round(daily_calories, 2)
    })

class HealthToolkit:
    def __init__(self):
        self.bmi_tool = StructuredTool.from_function(
            func=compute_bmi,
            name="calculate_bmi",
            description="Calculate Body Mass Index (BMI) and category from weight in kg and height in cm",
            args_schema=BMIInput
        )
        self.calorie_tool = StructuredTool.from_function(
            func=compute_daily_calories,
            name="calculate_daily_calories",
            description="Calculate daily maintenance calorie needs using weight, height, age, and activity level",
            args_schema=DailyCalorieInput
        )

    def get_tools(self) -> list:
        return [self.bmi_tool, self.calorie_tool]

toolkit = HealthToolkit()
toolkit_tools = toolkit.get_tools()

tool_map = {tool.name: tool for tool in toolkit_tools}

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
llm_with_tools = llm.bind_tools(toolkit_tools)

user_query = "I am 70kg, 175cm tall, 25 years old, and moderately active. What is my BMI and daily calorie need?"
messages = [HumanMessage(content=user_query)]

print("=== Exercise 8: Health Metrics Toolkit ===")
print("Toolkit Registered Tools:", [t.name for t in toolkit_tools])
print("User Query:", user_query)

round_counter = 1
while round_counter <= 4:
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    if not ai_response.tool_calls:
        print("\nFinal Model Health Metrics Summary:")
        print(ai_response.content)
        break

    for call in ai_response.tool_calls:
        print(f"\nRound {round_counter} - Model Called: {call['name']}")
        print(f"Arguments: {call['args']}")
        print(f"Call ID:   {call['id']}")

        selected_tool = tool_map[call["name"]]
        tool_output = selected_tool.invoke(call["args"])
        print("Executed Tool Output:")
        print(tool_output)

        messages.append(
            ToolMessage(
                content=str(tool_output),
                tool_call_id=call["id"]
            )
        )

    round_counter += 1
