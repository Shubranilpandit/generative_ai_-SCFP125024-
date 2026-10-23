"""
Lab 1 — Activity 8: Temperature Experiment
Author: Shubranil Pandit (SCFP125024)
Course: MCA 3rd Semester — Generative AI Lab

WHAT:
    This activity investigates the effect of the 'temperature' sampling hyperparameter
    in Generative LLMs. It submits the exact same prompt to the same model across three
    distinct temperature settings:
      - Low Temperature (0.1)
      - Medium Temperature (0.7)
      - High Temperature (1.0)
    It compares predictability, stylistic diversity, and token randomness.

WHY:
    Temperature is one of the most fundamental generation controls in LLM deployment.
    A common misconception is that temperature changes the model's factual knowledge or
    "makes the model smarter". In reality, temperature strictly modifies the Softmax
    probability distribution over vocabulary tokens during autoregressive sampling.
    Understanding when to dial temperature down (coding, math, structured data) versus
    up (creative writing, brainstorming) is an essential practical engineering skill.

HOW:
    - Sets a consistent prompt: "Complete this sentence with a creative metaphor in one sentence: A CPU is like..."
    - Queries the model at temperatures 0.1, 0.7, and 1.0 while holding all other generation
      parameters (max_tokens, top_p) strictly constant.
    - Uses Groq Cloud API (`qwen/qwen3.8-27b`) with graceful fallback to local Ollama.
    - Sanitizes unicode characters for reliable display on Windows terminals.
    - Formats observations and explains the underlying mathematical Softmax scaling formula:
        P(token_i) = exp(z_i / T) / sum_j(exp(z_j / T))

OBSERVATION:
    - Temperature 0.1 produced highly deterministic, focused, and predictable output
      (e.g., standard librarian or calculator analogies).
    - Temperature 0.7 produced balanced, engaging phrasing with moderate stylistic variety.
    - Temperature 1.0 produced more adventurous, poetic, and varied vocabulary (flattened distribution).
    - In all cases, the underlying factual role of a CPU (processing instructions) remained consistent.

CONCLUSION:
    Temperature affects probability sampling, not stored weights or model knowledge.
    Low temperature (0.0 - 0.2) is ideal for factual extraction and code, while high
    temperature (0.8 - 1.2) is optimal for creative and exploratory applications.
"""

import sys
import os
import time
from typing import Dict, Any, Optional

# Optional imports for API provider and local Ollama
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
        # Check local and parent .env
        for path in [os.path.join(os.path.dirname(__file__), ".env"),
                     os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))]:
            if os.path.exists(path):
                load_dotenv(path)
                key = os.getenv("GROQ_API_KEY")
                if key and len(key.strip()) > 5:
                    return key.strip()
    return None


def generate_with_api(client: Groq, model: str, prompt: str, temperature: float) -> Dict[str, Any]:
    """Generates completion using Groq Cloud API at a specified temperature."""
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=100
    )
    elapsed = time.time() - t0
    raw_text = response.choices[0].message.content.strip()
    clean_text = sanitize_for_console(raw_text)
    return {
        "text": clean_text,
        "latency_sec": round(elapsed, 2),
        "tokens": response.usage.completion_tokens if response.usage else 0
    }


def generate_with_ollama(model: str, prompt: str, temperature: float) -> Dict[str, Any]:
    """Generates completion using local Ollama instance at a specified temperature."""
    endpoint = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 100}
    }
    t0 = time.time()
    res = requests.post(endpoint, json=payload, timeout=60)
    elapsed = time.time() - t0
    data = res.json()
    raw_text = data.get("message", {}).get("content", "").strip()
    clean_text = sanitize_for_console(raw_text)
    return {
        "text": clean_text,
        "latency_sec": round(elapsed, 2),
        "tokens": data.get("eval_count", 0)
    }


def print_mathematical_explanation() -> None:
    """Explains the mathematical principle of Softmax temperature scaling."""
    print("=" * 80)
    print("THE MATHEMATICS OF TEMPERATURE SAMPLING")
    print("=" * 80)
    print("""
When an LLM generates the next token, the final linear layer produces a vector of
unnormalized log-odds (called 'logits', z_i) for every word in the vocabulary (V):

                   exp( z_i / T )
    P(token_i) = --------------------
                 sum_j [ exp( z_j / T ) ]

1. EFFECT OF LOW TEMPERATURE (T -> 0):
   - Dividing logits by a small number (e.g. T = 0.1) amplifies differences between logits.
   - The highest logit dwarfs all other candidates, causing its probability to approach 1.0.
   - The output becomes greedy, highly repetitive, and deterministic.

2. EFFECT OF HIGH TEMPERATURE (T >= 1.0):
   - Dividing logits by a larger number compresses the differences between logits.
   - The probability distribution flattens toward a uniform distribution.
   - Rare, unexpected, or lower-probability tokens get a higher chance of being sampled,
     resulting in high creative variation (or hallucinations if T > 1.5).

3. CRITICAL FACTOR:
   - Temperature does NOT alter the trained neural weights or facts stored in the model.
   - It is purely a post-processing mathematical filter applied during token selection!
""")


def print_theoretical_answers() -> None:
    """Answers the assignment questions for Lab 1 Activity 8."""
    print("=" * 80)
    print("ASSIGNMENT QUESTIONS & VIVA PREPARATION")
    print("=" * 80)
    print("""
Q1: Which output felt most predictable? Most creative?
A1:
    - Most Predictable: Temperature 0.1. The model selected the most statistically
      probable tokens, yielding a standard, conventional metaphor (e.g., comparing a CPU
      to an organized librarian or calculator). Repeating the prompt multiple times at
      T=0.1 produces almost identical results.
    - Most Creative: Temperature 1.0. The model sampled from a broader probability distribution,
      introducing richer vocabulary, poetic flourishes, and unexpected analogies
      (e.g., comparing a CPU to a musical conductor or pulsating electric heart).

Q2: Does temperature change what the model knows?
A2: ABSOLUTELY NOT. Temperature is strictly a decoding hyperparameter applied to the
    Softmax output layer. The model's neural weights, parameter values, training data,
    and semantic knowledge remain 100% frozen. It only alters the probability distribution
    from which the next token is sampled.

Q3: When would you want low temperature? High temperature?
A3:
    USE LOW TEMPERATURE (0.0 - 0.3) WHEN:
      - Writing code or SQL queries (syntax must be exact; hallucination is unacceptable).
      - Extracting structured data (e.g. JSON extraction from unstructured text).
      - Answering factual, technical, mathematical, or legal queries.
      - Classification and deterministic evaluation tasks.

    USE HIGH TEMPERATURE (0.7 - 1.2) WHEN:
      - Creative writing, storytelling, poetry, and dialogue.
      - Brainstorming novel product names, marketing copy, or business ideas.
      - Generating diverse synthetic training datasets.
      - Exploratory conversations where variety is prized over uniformity.
""")


def main() -> None:
    """Main execution point for Activity 8."""
    print("=" * 80)
    print("LAB 1: ACTIVITY 8 - TEMPERATURE EXPERIMENT")
    print("=" * 80)

    test_prompt = "Complete this sentence with a creative metaphor in one sentence: A CPU is like..."
    temperatures = [0.1, 0.7, 1.0]

    # Backend selection: Prefer Groq API for speed, fallback to local Ollama
    api_key = load_api_key()
    backend = None
    model_name = None
    client = None

    if GROQ_AVAILABLE and api_key:
        backend = "api"
        model_name = "qwen/qwen3.8-27b"
        client = Groq(api_key=api_key)
        print(f"Backend Selected: Hosted API (Groq Cloud)")
        print(f"Model ID        : {model_name}")
    elif REQUESTS_AVAILABLE:
        backend = "ollama"
        model_name = "qwen3:4b"
        print(f"Backend Selected: Local Ollama (http://127.0.0.1:11434)")
        print(f"Model ID        : {model_name}")
    else:
        print("[!] Error: Neither Groq API key nor local Ollama service is available.")
        print("    Configure 'Lab 1/.env' with GROQ_API_KEY or launch Ollama.")
        print_mathematical_explanation()
        print_theoretical_answers()
        return

    print(f"Fixed Prompt    : \"{test_prompt}\"")
    print(f"Testing Temps   : {temperatures}\n")
    print("=" * 80)
    print("EXPERIMENTAL EXECUTION RESULTS")
    print("=" * 80)

    results = []
    for temp in temperatures:
        print(f"\n--- Testing Temperature = {temp} ---")
        try:
            if backend == "api":
                res = generate_with_api(client, model_name, test_prompt, temp)
            else:
                res = generate_with_ollama(model_name, test_prompt, temp)

            print(f"  Response: \"{res['text']}\"")
            print(f"  Latency : {res['latency_sec']}s | Tokens: {res['tokens']}")
            results.append((temp, res["text"]))
        except Exception as exc:
            print(f"  [!] Execution failed for temperature {temp}: {exc}")

    # Summary table
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE TEMPERATURE COMPARISON SUMMARY")
    print("=" * 80)
    for temp, text in results:
        print(f"[Temp {temp:>3.1f}] : {text}")
    print()

    print_mathematical_explanation()
    print_theoretical_answers()
    print("Activity 8 execution complete.")


if __name__ == "__main__":
    main()
