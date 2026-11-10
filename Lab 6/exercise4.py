import os
import sys
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

weather_database = {
    "mumbai": {"temperature_celsius": 32.5, "conditions": "humid and sunny"},
    "delhi": {"temperature_celsius": 18.0, "conditions": "chilly and foggy"},
    "bangalore": {"temperature_celsius": 24.0, "conditions": "pleasant and breezy"},
    "pune": {"temperature_celsius": 26.0, "conditions": "clear sky"},
}

@tool(description="Get the current temperature in Celsius and atmospheric conditions for a given city")
def get_weather(city: str) -> str:
    cleaned_city = city.strip().lower()
    data = weather_database.get(cleaned_city, {"temperature_celsius": 28.0, "conditions": "clear"})
    return json.dumps({
        "city": city.title(),
        "temperature_celsius": data["temperature_celsius"],
        "conditions": data["conditions"]
    })

@tool(description="Recommend a suitable outfit given the temperature in Celsius and weather conditions")
def recommend_outfit(temperature: float, conditions: str) -> str:
    cond_lower = conditions.lower()
    if temperature >= 30.0:
        base = "Lightweight breathable cotton or linen clothes, sunglasses, and a hat."
    elif temperature >= 20.0:
        base = "Comfortable casual wear such as a t-shirt or polo with light trousers."
    else:
        base = "Warm layers including a light jacket, sweater, or hoodie."

    if "rain" in cond_lower or "humid" in cond_lower:
        addons = "Carry an umbrella and wear water-friendly footwear."
    elif "sun" in cond_lower:
        addons = "Apply sunscreen and stay hydrated."
    elif "chilly" in cond_lower or "fog" in cond_lower:
        addons = "Wear a scarf and closed-toe shoes."
    else:
        addons = "Dress comfortably for normal weather."

    return f"Recommended outfit: {base} Note: {addons}"

tool_map = {
    get_weather.name: get_weather,
    recommend_outfit.name: recommend_outfit
}

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
llm_with_tools = llm.bind_tools([get_weather, recommend_outfit])

user_query = "What should I wear today in Mumbai? Please check the weather first and then recommend an outfit."
messages = [HumanMessage(content=user_query)]

print("=== Exercise 4: Weather-Based Outfit Recommendation ===")
print("User Query:", user_query)

round_counter = 1
while round_counter <= 4:
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    if not ai_response.tool_calls:
        print("\nFinal Model Recommendation:")
        print(ai_response.content)
        break

    print(f"\n--- Round {round_counter}: Model Tool Calls ---")
    for call in ai_response.tool_calls:
        print(f"Tool: {call['name']}")
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
