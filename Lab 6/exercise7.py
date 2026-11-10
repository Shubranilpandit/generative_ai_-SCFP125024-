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

@tool(description="Get the estimated round-trip flight cost between origin and destination cities")
def get_flight_cost(origin: str, destination: str) -> str:
    route = f"{origin.strip().lower()}-{destination.strip().lower()}"
    costs = {
        "delhi-goa": 9500.0,
        "mumbai-goa": 5500.0,
        "bangalore-goa": 6200.0
    }
    price = costs.get(route, 8500.0)
    return json.dumps({
        "origin": origin.title(),
        "destination": destination.title(),
        "estimated_flight_cost_inr": price,
        "trip_type": "Round trip per person"
    })

@tool(description="Get the estimated total hotel lodging cost for a given city and number of nights")
def get_hotel_cost(city: str, nights: int) -> str:
    nightly_rates = {
        "goa": 3500.0,
        "delhi": 4000.0,
        "mumbai": 5000.0
    }
    rate = nightly_rates.get(city.strip().lower(), 3000.0)
    total_lodging = rate * nights
    return json.dumps({
        "city": city.title(),
        "nights": nights,
        "rate_per_night_inr": rate,
        "total_hotel_cost_inr": total_lodging
    })

@tool(description="Get the estimated food and dining budget for a given city and number of days")
def get_food_budget(city: str, days: int) -> str:
    daily_rates = {
        "goa": 1500.0,
        "delhi": 1200.0,
        "mumbai": 1800.0
    }
    rate = daily_rates.get(city.strip().lower(), 1400.0)
    total_food = rate * days
    return json.dumps({
        "city": city.title(),
        "days": days,
        "daily_food_budget_inr": rate,
        "total_food_budget_inr": total_food
    })

tool_map = {
    get_flight_cost.name: get_flight_cost,
    get_hotel_cost.name: get_hotel_cost,
    get_food_budget.name: get_food_budget
}

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
llm_with_tools = llm.bind_tools([get_flight_cost, get_hotel_cost, get_food_budget])

user_query = "I am planning a 3-night trip from Delhi to Goa. What will flights, hotel, and food roughly cost?"
messages = [HumanMessage(content=user_query)]

print("=== Exercise 7: Trip Cost Estimator ===")
print("User Query:", user_query)

step_counter = 1
while step_counter <= 5:
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    if not ai_response.tool_calls:
        print("\nFinal Combined Travel Cost Estimate:")
        print(ai_response.content)
        break

    for call in ai_response.tool_calls:
        print(f"\nStep {step_counter} - Model Called Tool: {call['name']}")
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

    step_counter += 1
