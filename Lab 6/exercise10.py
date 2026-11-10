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

@tool(description="Search for restaurant names matching a specific cuisine in a given city")
def find_restaurants(cuisine: str, city: str) -> str:
    database = {
        ("italian", "pune"): ["Little Italy", "Alto Vino", "Le Plaisir"],
        ("chinese", "pune"): ["Mainland China", "Crazy Noodles"],
        ("italian", "mumbai"): ["Trattoria", "Celini"]
    }
    cleaned_cuisine = cuisine.strip().lower()
    cleaned_city = city.strip().lower()
    matches = database.get((cleaned_cuisine, cleaned_city), ["Little Italy", "Alto Vino"])
    return json.dumps({
        "cuisine": cuisine.title(),
        "city": city.title(),
        "restaurants": matches
    })

@tool(description="Look up the customer review rating out of 5 for a restaurant")
def get_restaurant_rating(restaurant_name: str) -> str:
    ratings = {
        "little italy": {"rating": 4.6, "reviews": 1280},
        "alto vino": {"rating": 4.8, "reviews": 940},
        "le plaisir": {"rating": 4.7, "reviews": 2100}
    }
    clean = restaurant_name.strip().lower()
    data = ratings.get(clean, {"rating": 4.4, "reviews": 500})
    return json.dumps({
        "restaurant_name": restaurant_name.title(),
        "rating_out_of_5": data["rating"],
        "total_reviews": data["reviews"]
    })

@tool(description="Check whether a restaurant is open on a specified day of the week")
def check_restaurant_open(restaurant_name: str, day: str) -> str:
    closed_days = {
        "little italy": ["monday"],
        "alto vino": [],
        "le plaisir": ["tuesday"]
    }
    clean_name = restaurant_name.strip().lower()
    clean_day = day.strip().lower()
    closed = closed_days.get(clean_name, [])
    is_open = clean_day not in closed
    return json.dumps({
        "restaurant_name": restaurant_name.title(),
        "day": day.title(),
        "is_open": is_open,
        "timings": "12:00 PM to 11:30 PM" if is_open else "Closed"
    })

@tool(description="Estimate the average dining cost for two people in INR at a restaurant")
def estimate_cost_for_two(restaurant_name: str) -> str:
    pricing = {
        "little italy": 1800.0,
        "alto vino": 3200.0,
        "le plaisir": 1500.0
    }
    clean_name = restaurant_name.strip().lower()
    approx_cost = pricing.get(clean_name, 2000.0)
    return json.dumps({
        "restaurant_name": restaurant_name.title(),
        "cost_for_two_inr": approx_cost
    })

all_tools = [
    find_restaurants,
    get_restaurant_rating,
    check_restaurant_open,
    estimate_cost_for_two
]

tool_map = {tool.name: tool for tool in all_tools}

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
llm_with_tools = llm.bind_tools(all_tools)

user_query = "Suggest a good Italian restaurant in Pune that is open today on Friday, with its rating and rough cost for two"
messages = [HumanMessage(content=user_query)]

print("=== Exercise 10: Restaurant Recommendation Assistant ===")
print("All Bound Tools:", [t.name for t in all_tools])
print("User Query:", user_query)

step_counter = 1
while step_counter <= 6:
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    if not ai_response.tool_calls:
        print("\nFinal Recommended Restaurant Summary:")
        print(ai_response.content)
        break

    for call in ai_response.tool_calls:
        print(f"\nStep {step_counter} - Model Invoked: {call['name']}")
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
