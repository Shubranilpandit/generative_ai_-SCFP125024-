import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

orders_database = {
    "ORD-4521": "Shipped - In transit, estimated delivery today by 5:00 PM via BlueDart",
    "ORD-1089": "Processing - Order packed at warehouse, awaiting courier pickup",
    "ORD-7740": "Delivered - Successfully handed over to recipient on November 8, 2026",
}

@tool(description="Look up the shipping and delivery status of an order using its order ID")
def lookup_order_status(order_id: str) -> str:
    cleaned_id = order_id.strip().upper()
    return orders_database.get(
        cleaned_id,
        f"Order {cleaned_id} was not found in the tracking database."
    )

tool_map = {
    lookup_order_status.name: lookup_order_status
}

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
llm_with_tools = llm.bind_tools([lookup_order_status])

user_query = "Where is my order ORD-4521?"
messages = [HumanMessage(content=user_query)]

print("=== Exercise 1: Order Status Lookup Tool ===")
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
            tool_call_id=call["id"],
        )
    )

final_response = llm_with_tools.invoke(messages)
print("\nFinal Model Answer:")
print(final_response.content)
