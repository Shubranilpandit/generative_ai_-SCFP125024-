"""
Lab 1 — Activity 5: Run an LLM Locally Using Ollama
Author: Shubranil Pandit (SCFP125024)
Course: MCA 3rd Semester — Generative AI Lab

WHAT:
    This activity demonstrates running an open-weight Large Language Model entirely
    on local hardware using Ollama's local REST API (http://localhost:11434).
    It queries the local service status, lists locally installed models, and performs
    an on-device text generation while profiling execution latency.

WHY:
    Running models locally provides complete data sovereignty and zero cloud operational
    costs. However, local inference requires careful management of compute and RAM.
    Understanding how local runtimes (such as Ollama and llama.cpp) execute inference,
    handle offline requirements, and manage memory constraints is an essential skill.

HOW:
    - Queries the Ollama local HTTP API (`/api/tags`) to discover installed models.
    - Sends an inference request to `/api/chat` using Python's `requests` library.
    - Measures end-to-end execution latency on local hardware.
    - Provides graceful error handling and startup guidance if the Ollama service is inactive.
    - Answers the 4 core assignment questions regarding internet access, compute location,
      privacy/telemetry, and memory exhaustion.

OBSERVATION:
    - The local Ollama service is active with model `qwen3:4b` installed (~2.5 GB INT4 quantized).
    - Inference executes locally on CPU cores and system RAM without an active internet connection.
    - CPU generation generates ~3.5 to 5.0 tokens per second due to memory bandwidth limits,
      compared to faster GPU speeds.

CONCLUSION:
    Local LLM execution with Ollama is completely private and functional offline once downloaded.
    However, generation speed on consumer laptops is bounded by CPU memory bandwidth, making
    quantization (INT4/INT8) and modest model sizes (1B to 4B) essential for acceptable latency.
"""

import sys
import os
import time
from typing import List, Dict, Any

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# Default Ollama host endpoint (can be overridden via environment variable)
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


def check_ollama_status(host: str = DEFAULT_OLLAMA_HOST) -> Dict[str, Any]:
    """
    Checks whether the local Ollama daemon/service is reachable and lists installed models.
    """
    result = {
        "reachable": False,
        "host": host,
        "models": [],
        "error_message": None
    }

    if not REQUESTS_AVAILABLE:
        result["error_message"] = "'requests' library is not installed in the environment."
        return result

    try:
        response = requests.get(f"{host}/api/tags", timeout=4)
        if response.status_code == 200:
            result["reachable"] = True
            data = response.json()
            models_list = data.get("models", [])
            for m in models_list:
                result["models"].append({
                    "name": m.get("name"),
                    "size_gb": round(m.get("size", 0) / (1024 ** 3), 2),
                    "modified_at": m.get("modified_at", "Unknown")[:10]
                })
        else:
            result["error_message"] = f"HTTP Error {response.status_code} received from Ollama host."
    except requests.exceptions.ConnectionError:
        result["error_message"] = (
            f"Could not connect to Ollama at {host}.\n"
            "   Ensure the Ollama application is running (e.g., start Ollama from the Start Menu\n"
            "   or execute 'ollama serve' in a separate terminal)."
        )
    except Exception as exc:
        result["error_message"] = str(exc)

    return result


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


def run_local_inference(model_name: str, prompt: str, host: str = DEFAULT_OLLAMA_HOST) -> Dict[str, Any]:
    """
    Executes a chat completion request against the local Ollama instance.
    Uses /api/chat which cleanly separates reasoning/thinking from the final response.
    """
    endpoint = f"{host}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    print(f"Sending prompt to local model '{model_name}'...")
    print(f"Prompt: \"{prompt}\"")
    print("Waiting for local CPU/GPU inference (no external network call)...")

    start_time = time.time()
    try:
        # 90-second timeout allows sufficient headroom for CPU-based local generation
        response = requests.post(endpoint, json=payload, timeout=90)
        elapsed_sec = round(time.time() - start_time, 2)

        if response.status_code == 200:
            data = response.json()
            msg = data.get("message", {})
            content = sanitize_for_console(msg.get("content", "").strip())
            thinking = sanitize_for_console(msg.get("thinking", "").strip())

            return {
                "success": True,
                "latency_sec": elapsed_sec,
                "content": content,
                "thinking": thinking,
                "eval_count": data.get("eval_count", 0),
                "eval_duration_sec": round(data.get("eval_duration", 0) / 1e9, 2)
            }
        else:
            return {
                "success": False,
                "latency_sec": elapsed_sec,
                "error": f"Ollama returned HTTP status {response.status_code}: {response.text}"
            }
    except requests.exceptions.Timeout:
        elapsed_sec = round(time.time() - start_time, 2)
        return {
            "success": False,
            "latency_sec": elapsed_sec,
            "error": (
                f"Inference request timed out after {elapsed_sec} seconds.\n"
                "   Explanation: On a standard laptop without a dedicated GPU, CPU generation for 4B models\n"
                "   runs at ~4-6 tokens/sec. Long prompts or extensive internal reasoning can take over a minute.\n"
                "   Recommendation: Use concise prompts or consider smaller models (e.g. 1.5B) for faster CPU turnaround."
            )
        }
    except requests.exceptions.ConnectionError:
        elapsed_sec = round(time.time() - start_time, 2)
        return {
            "success": False,
            "latency_sec": elapsed_sec,
            "error": (
                f"Connection dropped to Ollama host at {host}.\n"
                "   Ensure the Ollama background daemon did not terminate or run out of memory."
            )
        }
    except Exception as exc:
        elapsed_sec = round(time.time() - start_time, 2)
        return {
            "success": False,
            "latency_sec": elapsed_sec,
            "error": f"Unexpected error during local inference: {exc}"
        }


def display_ollama_installation_guide() -> None:
    """Prints instructions for installing Ollama and downloading models."""
    print("=" * 80)
    print("OLLAMA SETUP & MODEL DOWNLOAD GUIDE")
    print("=" * 80)
    print("""
Step 1: Download and install Ollama for Windows from the official website:
        https://ollama.com/download/windows

Step 2: Verify installation in PowerShell:
        ollama --version

Step 3: Pull a lightweight, student-friendly model (e.g., 1.5B to 4B parameters):
        ollama pull qwen3:4b
        # Or a smaller 1.5B model for very low RAM laptops:
        ollama pull qwen2.5:1.5b

Step 4: Run a quick command-line test:
        ollama run qwen3:4b "Explain gravity in one sentence."
""")


def print_theoretical_answers() -> None:
    """Answers the four core questions from Lab 1 Activity 5."""
    print("\n" + "=" * 80)
    print("THEORETICAL ANALYSIS & ASSIGNMENT QUESTIONS")
    print("=" * 80)
    print("""
Q1: Does the model need internet after downloading?
A1: NO. Once the model weights and GGUF manifest are downloaded to your disk
    (stored in ~/.ollama/models), inference is 100% offline. You can disconnect Wi-Fi
    completely or run on an isolated airplane network, and local inference functions
    without interruption.

Q2: Where is inference actually happening?
A2: Inference is executing directly on your computer's local hardware:
    - If a compatible NVIDIA/AMD GPU is present: The weights are loaded into GPU VRAM,
      and tensor operations run on CUDA / ROCm cores.
    - If no discrete GPU is present (standard laptop): Ollama executes matrix operations
      on your CPU cores (using AVX2/AVX-512 SIMD vector instructions) using system RAM.

Q3: Is any data leaving your machine?
A3: NO. During local inference, prompts, embeddings, and generated responses never
    leave your local machine. They are transmitted solely over the local loopback
    network interface (127.0.0.1:11434). Unlike cloud APIs, no prompt logging,
    external telemetry, or cloud retention occurs.

Q4: What happens if the model is too large for your memory?
A4: Two failure modes occur:
    1. Allocation Failure / OOM: If the system cannot allocate contiguous buffers for
       the model weights, the process crashes with an Out-Of-Memory (OOM) error or
       Ollama refuses to load the model.
    2. Severe Disk Swapping (Thrashing): If Windows attempts to keep the model alive by
       swapping memory pages to the disk paging file (pagefile.sys on SSD/HDD), token
       generation latency crashes from ~5 tokens/sec to less than 0.1 tokens/sec,
       causing the entire laptop UI to freeze and stutter.
""")


def main() -> None:
    """Main execution point for Activity 5."""
    print("=" * 80)
    print("LAB 1: ACTIVITY 5 - RUN AN LLM LOCALLY USING OLLAMA")
    print("=" * 80)

    # 1. Check local service reachability
    print(f"Connecting to local Ollama service at {DEFAULT_OLLAMA_HOST}...")
    status = check_ollama_status(DEFAULT_OLLAMA_HOST)

    if not status["reachable"]:
        print(f"\n[!] Status: Ollama service is currently unavailable.")
        print(f"    Reason: {status['error_message']}\n")
        display_ollama_installation_guide()
        print_theoretical_answers()
        return

    print("[+] Ollama service is LIVE and responding!")
    print(f"    Available Local Models ({len(status['models'])} found):")
    for idx, m in enumerate(status["models"]):
        print(f"      {idx + 1}. {m['name']:<18} | Size: ~{m['size_gb']} GB | Downloaded: {m['modified_at']}")

    # 2. Select model and run test inference
    if status["models"]:
        selected_model = status["models"][0]["name"]
        print(f"\nSelecting model '{selected_model}' for local inference demonstration...")
        test_prompt = "What is the primary difference between training and inference in Generative AI? Answer in 2 concise sentences."

        result = run_local_inference(selected_model, test_prompt)
        print("\n" + "-" * 80)
        print("LOCAL INFERENCE RESULT:")
        print("-" * 80)
        if result["success"]:
            print(f"Status      : Success (200 OK)")
            print(f"Total Time  : {result['latency_sec']} seconds")
            if result.get("eval_count") and result.get("eval_duration_sec"):
                speed = result["eval_count"] / result["eval_duration_sec"]
                print(f"Token Count : {result['eval_count']} tokens")
                print(f"Speed       : ~{speed:.1f} tokens/second")
            if result.get("thinking"):
                print(f"\nModel Internal Scratchpad Preview:")
                print(f"  \"{result['thinking'][:180]}...\"")
            print(f"\nGenerated Response:\n{result['content']}")
        else:
            print(f"Execution Error: {result['error']}")
    else:
        print("\n[!] No models are currently downloaded in Ollama.")
        print("    Run 'ollama pull qwen3:4b' to download a lightweight model.")

    # 3. Print theoretical answers
    print_theoretical_answers()
    print("Activity 5 execution complete.")


if __name__ == "__main__":
    main()
