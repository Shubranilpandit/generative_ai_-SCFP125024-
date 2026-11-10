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

class ExpenseSplitInput(BaseModel):
    total_amount: float = Field(gt=0, description="Total expense or bill amount to split")
    num_people: int = Field(gt=0, description="Number of people sharing the expense")

def split_expense(total_amount: float, num_people: int) -> str:
    per_person = round(total_amount / num_people, 2)
    result = {
        "total_amount": round(total_amount, 2),
        "num_people": num_people,
        "amount_per_person": per_person
    }
    return json.dumps(result)

splitter_tool = StructuredTool.from_function(
    func=split_expense,
    name="split_expense",
    description="Splits a total bill amount equally among a given number of people",
    args_schema=ExpenseSplitInput
)

tool_map = {
    splitter_tool.name: splitter_tool
}

print("=== Exercise 6: Expense Splitter Tool ===")

print("\n1. Direct Manual Tool Invocation:")
manual_input = {"total_amount": 2400.0, "num_people": 4}
manual_output = splitter_tool.invoke(manual_input)
print("Manual Input:", manual_input)
print("Manual Output:", manual_output)

print("\n2. Bound Model Tool Calling Flow:")
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
llm_with_tools = llm.bind_tools([splitter_tool])

user_query = "Split a bill of 2400 rupees equally between 4 friends"
print("User Query:", user_query)

messages = [HumanMessage(content=user_query)]
first_response = llm_with_tools.invoke(messages)
messages.append(first_response)

print("\nModel Suggested Tool Calls:")
bound_output = ""
for call in first_response.tool_calls:
    print(f"Tool Name: {call['name']}")
    print(f"Arguments: {call['args']}")
    print(f"Call ID:   {call['id']}")

    selected_tool = tool_map[call["name"]]
    bound_output = selected_tool.invoke(call["args"])
    print("\nManually Executed Tool Result from Bound Flow:")
    print(bound_output)

    messages.append(
        ToolMessage(
            content=str(bound_output),
            tool_call_id=call["id"]
        )
    )

final_response = llm_with_tools.invoke(messages)
print("\nFinal Model Answer:")
print(final_response.content)

print("\n3. Comparison and Verification:")
manual_data = json.loads(manual_output)
bound_data = json.loads(bound_output)
matches = manual_data == bound_data
print("Manual Calculation Data:    ", manual_data)
print("Bound Flow Calculation Data:", bound_data)
print("Do Both Approaches Match?    ", matches)
