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

@tool(description="Check whether a person has available calendar slots on a specific date")
def check_calendar_availability(person: str, date: str) -> str:
    cleaned_person = person.strip().title()
    available_calendar = {
        "Alex": ["2026-11-15", "2026-11-16", "2026-11-18"],
        "Priya": ["2026-11-14", "2026-11-15"]
    }
    dates = available_calendar.get(cleaned_person, ["2026-11-15"])
    is_free = date in dates
    return json.dumps({
        "person": cleaned_person,
        "date": date,
        "is_available": is_free,
        "available_slots": ["10:00 AM", "02:00 PM", "04:30 PM"] if is_free else []
    })

@tool(description="Get the standard UTC timezone offset and abbreviation for a city")
def get_timezone_offset(city: str) -> str:
    timezone_map = {
        "london": {"timezone": "GMT", "utc_offset": "+00:00", "standard": "UTC+0"},
        "new york": {"timezone": "EST", "utc_offset": "-05:00", "standard": "UTC-5"},
        "mumbai": {"timezone": "IST", "utc_offset": "+05:30", "standard": "UTC+5:30"},
        "tokyo": {"timezone": "JST", "utc_offset": "+09:00", "standard": "UTC+9"}
    }
    cleaned_city = city.strip().lower()
    data = timezone_map.get(cleaned_city, {"timezone": "UTC", "utc_offset": "+00:00", "standard": "UTC+0"})
    return json.dumps({"city": city.title(), **data})

@tool(description="Book and confirm a meeting with a person on a specific date and time")
def book_meeting(person: str, date: str, time: str) -> str:
    return json.dumps({
        "booking_id": "MTG-90214",
        "status": "Confirmed",
        "attendee": person.title(),
        "date": date,
        "time": time,
        "calendar_invite_sent": True
    })

tool_map = {
    check_calendar_availability.name: check_calendar_availability,
    get_timezone_offset.name: get_timezone_offset,
    book_meeting.name: book_meeting
}

tools_list = [check_calendar_availability, get_timezone_offset, book_meeting]

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
llm_with_tools = llm.bind_tools(tools_list)

def execute_chat_query(query_text: str, query_title: str):
    print(f"\n--- {query_title} ---")
    print("User Query:", query_text)
    messages = [HumanMessage(content=query_text)]
    step = 1
    called_tools = []

    while step <= 4:
        ai_response = llm_with_tools.invoke(messages)
        messages.append(ai_response)

        if not ai_response.tool_calls:
            print("\nFinal Model Answer:")
            print(ai_response.content)
            break

        for call in ai_response.tool_calls:
            called_tools.append(call["name"])
            print(f"Step {step} - Model Called Tool: {call['name']}")
            print(f"Arguments: {call['args']}")

            selected_tool = tool_map[call["name"]]
            tool_output = selected_tool.invoke(call["args"])
            print(f"Tool Output: {tool_output}")

            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=call["id"]
                )
            )

        step += 1

    return called_tools

print("=== Exercise 9: Smart Meeting Scheduler ===")
print("All Bound Tools:", [t.name for t in tools_list])

query_1 = "Is Alex available for a meeting on 2026-11-15?"
calls_1 = execute_chat_query(query_1, "Query 1: Availability Check Only")

query_2 = "Check if Alex is available on 2026-11-15, get the timezone offset for London, and book a meeting with Alex on 2026-11-15 at 14:00."
calls_2 = execute_chat_query(query_2, "Query 2: Full Booking Request")

print("\n=== Partial Tool Use Comparison ===")
print(f"Query 1 Tool Calls (Availability only): {calls_1}")
print(f"Query 2 Tool Calls (Full booking):      {calls_2}")
