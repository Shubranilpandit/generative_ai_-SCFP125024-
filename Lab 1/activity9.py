"""
Lab 1 — Activity 9: Build a Mini Chatbot
Author: Shubranil Pandit (SCFP125024)
Course: MCA 3rd Semester — Generative AI Lab

WHAT:
    This activity constructs a multi-turn conversational chatbot from scratch.
    It demonstrates how conversation history (a list of user and assistant message dicts)
    is maintained across turns and re-sent to the model on every iteration to provide
    contextual continuity.

WHY:
    Large Language Models are inherently STATELESS. Every API or inference call is an
    isolated, independent mathematical event. The model does not retain memory, learn,
    or remember who you are between requests. To create a coherent "chatbot" experience,
    the client application must manage conversation state by accumulating historical
    exchanges and passing them inside the model's context window.

HOW:
    - Initializes a stateful conversation history list: `messages = []`
    - Appends user inputs (`{"role": "user", "content": user_text}`)
    - Passes accumulated history to the inference backend (Groq API or local Ollama).
    - Appends model responses (`{"role": "assistant", "content": assistant_text}`)
    - Supports interactive command-line chat with clean exit commands (`exit`, `quit`).
    - Provides a built-in automated multi-turn verification mode (`--demo`) for automated testing.
    - Demonstrates what happens when history is dropped.

OBSERVATION:
    - When history is retained, the model correctly answers follow-up queries (e.g., Turn 1:
      "My name is Shubranil." -> Turn 2: "What is my name?" -> Answer: "Your name is Shubranil.").
    - When history is dropped or cleared, the model loses all context and cannot identify
      the user or reference previous statements.

CONCLUSION:
    The conversational memory of an LLM is an illusion created by the client application.
    Continuity is entirely dependent on re-submitting prior dialogue turns within the token limits
    of the model's context window.
"""

import sys
import os
import time
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    from groq import Groq, APIConnectionError, RateLimitError, AuthenticationError
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def sanitize_for_console(text: str) -> str:
    """Replaces unicode punctuation with ASCII equivalents for safe Windows terminal printing."""
    replacements = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
        "\u2026": "..."
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def load_api_key() -> Optional[str]:
    """Safely retrieves the GROQ_API_KEY from environment variables or .env."""
    key = os.getenv("GROQ_API_KEY")
    if key and len(key.strip()) > 5:
        return key.strip()

    if DOTENV_AVAILABLE:
        for path in [os.path.join(os.path.dirname(__file__), ".env"),
                     os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))]:
            if os.path.exists(path):
                load_dotenv(path)
                key = os.getenv("GROQ_API_KEY")
                if key and len(key.strip()) > 5:
                    return key.strip()
    return None


class MiniChatbot:
    """
    A lightweight, student-level chatbot maintaining conversational state.
    """
    def __init__(self, backend: str = "api", model_name: str = "qwen/qwen3.8-27b"):
        self.backend = backend
        self.model_name = model_name
        self.conversation_history: List[Dict[str, str]] = []
        self.api_client: Optional[Groq] = None
        self.ollama_endpoint = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434") + "/api/chat"

        if backend == "api":
            api_key = load_api_key()
            if api_key and GROQ_AVAILABLE:
                self.api_client = Groq(api_key=api_key)
            else:
                # Fallback to Ollama if API key unavailable
                self.backend = "ollama"
                self.model_name = "qwen3:4b"

    def send_message(self, user_text: str) -> str:
        """
        Appends user input, sends full accumulated history to the model,
        and appends the model's reply to preserve context.
        """
        # 1. Append user turn
        self.conversation_history.append({"role": "user", "content": user_text})

        # 2. Call backend with full history
        reply_text = ""
        try:
            if self.backend == "api" and self.api_client is not None:
                response = self.api_client.chat.completions.create(
                    model=self.model_name,
                    messages=self.conversation_history,
                    temperature=0.7,
                    max_tokens=200
                )
                reply_text = response.choices[0].message.content.strip()
            else:
                # Local Ollama fallback
                payload = {
                    "model": self.model_name,
                    "messages": self.conversation_history,
                    "stream": False,
                    "options": {"num_predict": 120}
                }
                res = requests.post(self.ollama_endpoint, json=payload, timeout=60)
                if res.status_code == 200:
                    reply_text = res.json().get("message", {}).get("content", "").strip()
                else:
                    reply_text = f"[Ollama Error {res.status_code}]"

        except Exception as exc:
            reply_text = f"[Error generating response: {exc}]"

        # Sanitize text for Windows display
        clean_reply = sanitize_for_console(reply_text)

        # 3. Append assistant response to preserve history
        self.conversation_history.append({"role": "assistant", "content": clean_reply})
        return clean_reply

    def clear_history(self) -> None:
        """Empties the conversation history to simulate state loss."""
        self.conversation_history.clear()


def run_automated_demonstration(bot: MiniChatbot) -> None:
    """
    Executes a multi-turn conversation test proving that:
      1. History enables context retention.
      2. Dropping history causes complete amnesia.
    """
    print("=" * 80)
    print("AUTOMATED MULTI-TURN CONVERSATION EXPERIMENT")
    print("=" * 80)

    # Turn 1: Introduce user identity
    print("\n[Turn 1]")
    user_msg_1 = "Hello! My name is Shubranil and I am studying Generative AI."
    print(f"User     : {user_msg_1}")
    reply_1 = bot.send_message(user_msg_1)
    print(f"Assistant: {reply_1}")

    # Turn 2: Query identity relying on Turn 1 context
    print("\n[Turn 2 - With Conversation History]")
    user_msg_2 = "What is my name, and what subject am I studying?"
    print(f"User     : {user_msg_2}")
    reply_2 = bot.send_message(user_msg_2)
    print(f"Assistant: {reply_2}")

    # Turn 3: Simulate dropping history
    print("\n[Turn 3 - Context Dropped / Memory Cleared]")
    print("[!] Clearing conversation history buffer...")
    bot.clear_history()
    user_msg_3 = "What is my name?"
    print(f"User     : {user_msg_3}")
    reply_3 = bot.send_message(user_msg_3)
    print(f"Assistant: {reply_3}")


def print_theoretical_answers() -> None:
    """Answers the assignment questions for Lab 1 Activity 9."""
    print("\n" + "=" * 80)
    print("THEORETICAL ANALYSIS & ASSIGNMENT QUESTIONS")
    print("=" * 80)
    print("""
Q1: Why does the chatbot need to remember previous messages?
A1: Large Language Models are completely stateless functions (y = f(x)). The model
    does not store session memory, user profiles, or past conversation turns in its weights.
    Without the client sending the conversation history, the model sees each prompt in
    total isolation and cannot resolve pronouns ("it", "they"), follow instructions, or
    remember previous constraints.

Q2: What happens if you drop the conversation history?
A2: The chatbot suffers immediate, total amnesia. As demonstrated in Turn 3 above,
    if the history list is cleared or omitted, asking "What is my name?" fails because
    the model has no access to the previous turn where the name was introduced.

Q3: What's the difference between this and a single one-off prompt?
A3:
    - One-Off Prompt (Single-Turn): Sends a single text input and receives an isolated
      completion (e.g., "Translate this sentence to French"). No past context is needed.
    - Multi-Turn Chatbot: Sends an ever-growing array of message dictionaries:
      [
        {"role": "user", "content": "Hello..."},
        {"role": "assistant", "content": "Hi..."},
        {"role": "user", "content": "What did I just say?"}
      ]
      The model generates its next response by attending across the entire historical sequence.
""")


def main() -> None:
    """Main execution point for Activity 9."""
    print("=" * 80)
    print("LAB 1: ACTIVITY 9 - BUILD A MINI CHATBOT")
    print("=" * 80)

    # Initialize chatbot
    bot = MiniChatbot(backend="api", model_name="qwen/qwen3.8-27b")
    print(f"Active Backend: {bot.backend.upper()} (Model: {bot.model_name})")

    # Run automated demonstration first to verify history retention
    run_automated_demonstration(bot)

    # Print theoretical answers
    print_theoretical_answers()

    # Optional interactive loop if running in an interactive terminal
    if sys.stdin.isatty():
        print("=" * 80)
        print("INTERACTIVE CHAT MODE (Type 'exit' or 'quit' to end session)")
        print("=" * 80)
        bot.clear_history()
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Exiting chat. Goodbye!")
                    break
                reply = bot.send_message(user_input)
                print(f"Assistant: {reply}")
            except (KeyboardInterrupt, EOFError):
                print("\nSession terminated.")
                break
    else:
        print("\n(Non-interactive environment detected; interactive CLI prompt skipped).")

    print("\nActivity 9 execution complete.")


if __name__ == "__main__":
    main()
