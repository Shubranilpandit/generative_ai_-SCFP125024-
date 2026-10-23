"""
Lab 1 — Activity 6: Use an LLM Through an API / Inference Provider
Author: Shubranil Pandit (SCFP125024)
Course: MCA 3rd Semester — Generative AI Lab

WHAT:
    This activity demonstrates consuming a Large Language Model hosted on high-performance
    cloud infrastructure via a REST/SDK API (Groq Cloud). It authenticates using environment
    variables, sends an inference payload, parses the generated response and token usage metrics,
    and measures network and compute latency.

WHY:
    Most production applications do not host LLMs on local end-user devices; they interact
    with hosted cloud providers (such as Groq, OpenAI, Anthropic, or Azure).
    Understanding the client-server request cycle, authentication mechanisms, token-based
    billing models, and error handling (rate limits, network drops) is vital for any AI engineer.

HOW:
    - Loads the API key securely from environment variables (`GROQ_API_KEY`) using `python-dotenv`.
      (Checks local `Lab 1/.env` and parent workspace `.env`).
    - Never prints or exposes credentials in terminal outputs or logs.
    - Uses the official `groq` Python client library to dispatch chat completion requests.
    - Captures prompt tokens, completion tokens, latency, and throughput metrics.
    - Handles missing keys, authentication errors, and rate limits gracefully.
    - Answers the 3 core questions regarding compute ownership, request lifecycle, and trade-offs.

OBSERVATION:
    - Hosted API inference generated 172 tokens in ~0.85 seconds (~200+ tokens/second).
    - In comparison, local CPU inference in Activity 5 ran at ~5.4 tokens/second.
    - Hardware load on the local laptop was virtually zero (minimal CPU and RAM utilization).

CONCLUSION:
    Cloud API inference offloads all memory and compute burdens to specialized enterprise hardware
    (e.g., Groq LPUs or NVIDIA H100 clusters) delivering sub-second latency. However, it requires
    active internet connectivity, exposes data to cloud transit, and incurs per-token usage costs.
"""

import sys
import os
import time
from typing import Dict, Any, Optional

# Attempt to load python-dotenv and groq
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import groq
    from groq import Groq, APIConnectionError, RateLimitError, AuthenticationError, NotFoundError
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


def load_api_credentials() -> Optional[str]:
    """
    Safely searches for GROQ_API_KEY in environment variables and .env files.
    Checks both the local directory and parent workspace root.
    Never exposes or logs the actual secret key.
    """
    # 1. Check if already present in environment
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and len(api_key.strip()) > 5:
        return api_key.strip()

    # 2. Check local .env file
    if DOTENV_AVAILABLE:
        local_env = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(local_env):
            load_dotenv(local_env)
            api_key = os.getenv("GROQ_API_KEY")
            if api_key and len(api_key.strip()) > 5:
                return api_key.strip()

        # 3. Check parent workspace .env file
        parent_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
        if os.path.exists(parent_env):
            load_dotenv(parent_env)
            api_key = os.getenv("GROQ_API_KEY")
            if api_key and len(api_key.strip()) > 5:
                return api_key.strip()

    return None


def sanitize_for_console(text: str) -> str:
    """Replaces unicode typographic punctuation with ASCII equivalents for safe Windows terminal printing."""
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


def call_groq_inference(prompt: str, model_name: str = "openai/gpt-oss-20b") -> Dict[str, Any]:
    """
    Calls the Groq inference provider using the official Python SDK.
    Includes robust error handling for missing keys, network issues, and rate limits.
    """
    api_key = load_api_credentials()

    if not GROQ_AVAILABLE:
        return {
            "success": False,
            "error_type": "DependencyMissing",
            "message": "The 'groq' package is not installed. Run: pip install groq"
        }

    if not api_key:
        return {
            "success": False,
            "error_type": "MissingCredentials",
            "message": (
                "GROQ_API_KEY was not found in environment variables or .env files.\n"
                "To configure:\n"
                "  1. Obtain a free API key at https://console.groq.com\n"
                "  2. Add it to 'Lab 1/.env': GROQ_API_KEY=your_key_here\n"
                "  3. Ensure 'Lab 1/.env' is never committed to GitHub!"
            )
        }

    try:
        client = Groq(api_key=api_key)
        start_time = time.time()

        # Send Chat Completion request
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.7
        )

        total_elapsed = time.time() - start_time
        content = response.choices[0].message.content.strip()
        content = sanitize_for_console(content)

        # Extract usage statistics
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        tokens_per_sec = (completion_tokens / total_elapsed) if total_elapsed > 0 else 0.0

        return {
            "success": True,
            "model": model_name,
            "content": content,
            "total_elapsed_sec": round(total_elapsed, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "throughput_tokens_sec": round(tokens_per_sec, 1)
        }

    except AuthenticationError:
        return {
            "success": False,
            "error_type": "AuthenticationError",
            "message": "Authentication failed: The provided GROQ_API_KEY is invalid or revoked."
        }
    except NotFoundError as exc:
        return {
            "success": False,
            "error_type": "ModelNotFoundError",
            "message": f"Requested model '{model_name}' was not found: {exc.message}. Verify available models in your Groq account."
        }
    except RateLimitError:
        return {
            "success": False,
            "error_type": "RateLimitError",
            "message": "Rate limit exceeded: You have hit your provider's requests-per-minute (RPM) or tokens-per-minute (TPM) limit."
        }
    except APIConnectionError as exc:
        return {
            "success": False,
            "error_type": "ConnectionError",
            "message": f"Network error connecting to Groq API servers: {exc}"
        }
    except Exception as exc:
        return {
            "success": False,
            "error_type": "GenericError",
            "message": f"An unexpected error occurred during API inference: {exc}"
        }


def print_theoretical_answers() -> None:
    """Answers the core questions from Lab 1 Activity 6."""
    print("\n" + "=" * 80)
    print("THEORETICAL ANALYSIS & ASSIGNMENT QUESTIONS")
    print("=" * 80)
    print("""
Q1: What is actually happening between your program and the model?
A1: The complete client-server request cycle consists of 6 sequential steps:
    1. Serialization: Your Python program formats your prompt into a JSON payload.
    2. Network Transmission: The payload is encrypted (TLS/HTTPS) and transmitted over
       the public internet to the cloud provider's API Gateway.
    3. Authentication & Rate-Limiting: The gateway verifies your API key, checks your account
       tier and rate limits (tokens/min), and routes the request to an available worker.
    4. Hardware Inference: The prompt is loaded into high-speed VRAM/SRAM across specialized
       datacenter hardware (e.g. Groq LPUs or NVIDIA H100 GPUs) which execute the matrix
       multiplications in parallel.
    5. Response Packaging: The generated completion tokens, timing metrics, and usage statistics
       are serialized into a JSON response packet.
    6. Return & Parsing: The JSON response travels back over HTTPS to your local machine, where
       your Python script decodes and displays it.

Q2: Who is paying for compute here?
A2: The API customer (the account owner who registered the API key).
    - The provider (Groq, OpenAI, Anthropic) owns and operates the multi-million dollar data
      centers, paying for electricity, liquid cooling, server racks, and hardware depreciation.
    - In exchange, the customer pays on a metering model (typically priced per 1,000,000 tokens),
      or consumes a complimentary free-tier developer grant.

Q3: What are the trade-offs versus running locally?
A3:
    +------------------------+---------------------------------+---------------------------------+
    | Dimension              | Hosted API Inference (Groq)    | Local Inference (Ollama)        |
    +------------------------+---------------------------------+---------------------------------+
    | Local Hardware Burden  | Zero (Runs on any potato PC)    | High (Requires 8-16 GB+ RAM)    |
    | Inference Speed        | Extremely Fast (~200+ tok/s)    | Moderate on CPU (~3-6 tok/s)    |
    | Model Capability       | Access to 70B - 405B+ models    | Limited to 1B - 7B models       |
    | Privacy & Governance   | Prompts leave the machine       | 100% private; zero data transit |
    | Internet Requirement   | Strict (Fails if offline)       | Zero (Works 100% offline)       |
    | Cost Structure         | Pay-per-token recurring cost    | Free after hardware purchase    |
    +------------------------+---------------------------------+---------------------------------+
""")


def main() -> None:
    """Main execution point for Activity 6."""
    print("=" * 80)
    print("LAB 1: ACTIVITY 6 - USE AN LLM THROUGH AN API PROVIDER")
    print("=" * 80)

    # 1. Check API Key configuration
    api_key = load_api_credentials()
    if api_key:
        print("[+] API Key detected successfully in environment! (Key secret is hidden)")
    else:
        print("[-] Notice: No active API key found in environment variables.")

    # 2. Execute demonstration prompt
    target_model = "openai/gpt-oss-20b"
    sample_prompt = "What is the role of an inference engine in Generative AI? Answer in 2 clear sentences."

    print(f"\nDispatching API inference request to Groq Cloud...")
    print(f"Target Model: {target_model}")
    print(f"Prompt      : \"{sample_prompt}\"")

    result = call_groq_inference(sample_prompt, model_name=target_model)

    print("\n" + "-" * 80)
    print("API INFERENCE EXECUTION RESULT:")
    print("-" * 80)

    if result["success"]:
        print(f"Status           : Success (HTTP 200 OK)")
        print(f"Model Used       : {result['model']}")
        print(f"Total Roundtrip  : {result['total_elapsed_sec']} seconds")
        print(f"Prompt Tokens    : {result['prompt_tokens']}")
        print(f"Completion Tokens: {result['completion_tokens']}")
        print(f"Total Tokens     : {result['total_tokens']}")
        print(f"Throughput       : ~{result['throughput_tokens_sec']} tokens/second")
        print(f"\nGenerated Response:\n{result['content']}")
    else:
        print(f"Execution Status : Error ({result.get('error_type')})")
        print(f"Message          :\n{result.get('message')}")

    # 3. Print comprehensive theoretical answers
    print_theoretical_answers()
    print("Activity 6 execution complete.")


if __name__ == "__main__":
    main()
