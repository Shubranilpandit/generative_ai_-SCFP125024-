import os
import sys
import json
import asyncio
from typing import Type
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

inventory_database = {
    "P1023": {"product_name": "Wireless Ergonomic Mouse", "stock": 42, "status": "In Stock", "warehouse": "Bengaluru"},
    "P2045": {"product_name": "USB-C Mechanical Keyboard", "stock": 0, "status": "Out of Stock", "warehouse": "Pune"},
    "P3391": {"product_name": "27-inch 4K Monitor", "stock": 9, "status": "In Stock", "warehouse": "Mumbai"}
}

class ProductStockInput(BaseModel):
    product_id: str = Field(description="Unique alphanumeric ID of the product")

class ProductStockTool(BaseTool):
    name: str = "check_product_stock"
    description: str = "Checks inventory availability, warehouse location, and quantity for a product ID"
    args_schema: Type[BaseModel] = ProductStockInput

    def _run(self, product_id: str) -> str:
        clean_id = product_id.strip().upper()
        item = inventory_database.get(clean_id)
        if item:
            return json.dumps({"product_id": clean_id, **item})
        return json.dumps({"product_id": clean_id, "status": "Product ID not found in inventory"})

    async def _arun(self, product_id: str) -> str:
        await asyncio.sleep(0.05)
        return self._run(product_id=product_id)

stock_tool = ProductStockTool()

print("=== Exercise 5: Async Product Stock Checker ===")

print("\n1. Manual Synchronous Invocation (.invoke):")
sync_result = stock_tool.invoke({"product_id": "P1023"})
print("Result for P1023:", sync_result)

print("\n2. Manual Asynchronous Invocation (.ainvoke):")
async_result = asyncio.run(stock_tool.ainvoke({"product_id": "P2045"}))
print("Result for P2045:", async_result)

print("\n3. Bound LLM Natural Language Flow:")
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
llm_with_tools = llm.bind_tools([stock_tool])

user_query = "Is product P1023 in stock?"
print("User Query:", user_query)

messages = [HumanMessage(content=user_query)]
first_response = llm_with_tools.invoke(messages)
messages.append(first_response)

print("\nModel Suggested Tool Calls:")
for call in first_response.tool_calls:
    print(f"Tool Name: {call['name']}")
    print(f"Arguments: {call['args']}")
    print(f"Call ID:   {call['id']}")

for call in first_response.tool_calls:
    tool_output = stock_tool.invoke(call["args"])
    print("\nExecuted Tool Result from Bound Flow:")
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
