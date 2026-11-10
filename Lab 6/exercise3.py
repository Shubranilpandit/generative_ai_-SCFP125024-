import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

@tool(description="Convert temperature between Celsius and Fahrenheit units")
def convert_temperature(value: float, from_unit: str = "celsius", to_unit: str = "fahrenheit") -> str:
    unit_from = from_unit.strip().lower()
    if unit_from.startswith("c"):
        converted = (value * 9.0 / 5.0) + 32.0
        return f"{value} Celsius is equal to {round(converted, 2)} Fahrenheit"
    else:
        converted = (value - 32.0) * 5.0 / 9.0
        return f"{value} Fahrenheit is equal to {round(converted, 2)} Celsius"

@tool(description="Convert distance between kilometers and miles")
def convert_distance(value: float, from_unit: str = "km", to_unit: str = "miles") -> str:
    unit_from = from_unit.strip().lower()
    if unit_from in ["km", "kilometer", "kilometers"]:
        converted = value * 0.621371
        return f"{value} kilometers is equal to {round(converted, 2)} miles"
    else:
        converted = value / 0.621371
        return f"{value} miles is equal to {round(converted, 2)} kilometers"

tool_map = {
    convert_temperature.name: convert_temperature,
    convert_distance.name: convert_distance
}

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
llm_with_tools = llm.bind_tools([convert_temperature, convert_distance])

user_query = "Convert 100 km to miles"
messages = [HumanMessage(content=user_query)]

print("=== Exercise 3: Unit Converter Toolkit ===")
print("Registered Tools:", list(tool_map.keys()))
print("User Query:", user_query)

first_response = llm_with_tools.invoke(messages)
messages.append(first_response)

print("\nModel Suggested Tool Calls:")
for call in first_response.tool_calls:
    print(f"Tool Name: {call['name']}")
    print(f"Arguments: {call['args']}")
    print(f"Call ID:   {call['id']}")

for call in first_response.tool_calls:
    selected_tool = tool_map[call["name"]]
    tool_output = selected_tool.invoke(call["args"])
    print("\nManually Executed Tool Result:")
    print(tool_output)
    messages.append(
        ToolMessage(
            content=str(tool_output),
            tool_call_id=call["id"]
        )
    )

final_response = llm_with_tools.invoke(messages)
print("\nFinal Model Answer:")
print(final_response.content)
