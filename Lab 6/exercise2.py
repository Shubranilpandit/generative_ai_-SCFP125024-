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

class LoanEMIInput(BaseModel):
    principal: float = Field(gt=0, description="Total loan principal amount")
    annual_rate: float = Field(ge=0, le=30, description="Annual interest rate in percentage, between 0 and 30")
    tenure_months: int = Field(gt=0, description="Loan tenure duration in number of months")

def calculate_loan_emi(principal: float, annual_rate: float, tenure_months: int) -> str:
    if annual_rate == 0:
        monthly_emi = principal / tenure_months
        total_payment = principal
        total_interest = 0.0
    else:
        monthly_rate = (annual_rate / 100.0) / 12.0
        factor = (1.0 + monthly_rate) ** tenure_months
        monthly_emi = (principal * monthly_rate * factor) / (factor - 1.0)
        total_payment = monthly_emi * tenure_months
        total_interest = total_payment - principal

    result = {
        "principal": round(principal, 2),
        "annual_interest_rate_percent": round(annual_rate, 2),
        "tenure_months": tenure_months,
        "monthly_emi": round(monthly_emi, 2),
        "total_interest_payable": round(total_interest, 2),
        "total_amount_payable": round(total_payment, 2)
    }
    return json.dumps(result)

loan_emi_tool = StructuredTool.from_function(
    func=calculate_loan_emi,
    name="calculate_loan_emi",
    description="Calculate monthly EMI, total interest, and total payable amount for a loan with validation",
    args_schema=LoanEMIInput
)

tool_map = {
    loan_emi_tool.name: loan_emi_tool
}

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
llm_with_tools = llm.bind_tools([loan_emi_tool])

user_query = "What is my EMI for a 500000 loan at 9% annual interest for 24 months?"
messages = [HumanMessage(content=user_query)]

print("=== Exercise 2: Loan EMI Calculator Tool ===")
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
