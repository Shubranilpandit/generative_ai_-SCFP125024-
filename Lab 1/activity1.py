"""
Lab 1 — Activity 1: Explore LLM Models on Hugging Face
Author: Shubranil Pandit (SCFP125024)
Course: MCA 3rd Semester — Generative AI Lab

WHAT:
    This activity investigates three real language models on Hugging Face
    spanning small, medium, and large tiers:
      1. SmolLM2-135M-Instruct (Small)
      2. Llama-3.2-3B-Instruct (Medium)
      3. Llama-3.1-70B-Instruct (Large)
    It displays verified specifications, calculates theoretical memory
    footprints across precisions (FP16 vs INT4), and explains hardware feasibility.

WHY:
    Understanding model size, architecture, context window, and memory requirements
    is the critical first step in Generative AI engineering. Before selecting an LLM,
    a developer must determine whether the target hardware (laptop, GPU server, or cloud API)
    can accommodate the model's weights and KV cache.

HOW:
    - Stores verified metadata from official Hugging Face model cards.
    - Demonstrates optional live API verification using the Hugging Face Hub REST API
      (handling network availability gracefully).
    - Computes theoretical memory footprints using the formula:
        Weight Memory (Bytes) = Parameter Count * (Bits per Parameter / 8)
    - Distinguishes verified model specifications from deployment estimates.

OBSERVATION:
    - Small models (~135M) require less than 300 MB of RAM, running easily on standard CPUs.
    - Medium models (~3B) require ~6.4 GB in FP16, but become accessible on 8 GB laptops
      when quantized to INT4 (~1.6 - 2.2 GB).
    - Large models (~70B) demand ~140 GB in FP16 and ~35 GB in INT4, making local execution
      on consumer laptops impossible; they necessitate dedicated multi-GPU data center hardware.

CONCLUSION:
    Model parameter count and precision dictate minimum hardware requirements.
    Quantization democratizes local execution for 3B-8B parameter models on consumer laptops,
    while 70B+ models remain strictly in the domain of cloud APIs or enterprise GPU clusters.
"""

import sys
import json
from typing import Dict, Any

# Optional import for live metadata verification
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# Documented and verified model cards from Hugging Face
# Official links:
# 1. https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
# 2. https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
# 3. https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct
MODELS_CATALOG: Dict[str, Dict[str, Any]] = {
    "small": {
        "tier": "Small (~135 Million Parameters)",
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "creator": "Hugging Face (TB / SmolLM Team)",
        "parameter_count_exact": 134_515_200,
        "parameter_count_display": "135 Million (0.135B)",
        "is_generative": True,
        "model_type": "Causal Language Model (Transformer decoder)",
        "context_length_verified": 8192,
        "context_length_display": "8,192 tokens (8k)",
        "suitability_local_execution": "Highly Suitable (Any modern CPU, laptop, Raspberry Pi, or phone)",
        "recommended_hardware": "Standard CPU (x86 or ARM), 1 GB to 2 GB free RAM. GPU optional.",
        "huggingface_url": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct",
        "primary_use_cases": "On-device reasoning, summarization, edge computing, educational experiments",
        "license": "Apache-2.0 (Open source)"
    },
    "medium": {
        "tier": "Medium (~3.2 Billion Parameters)",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "creator": "Meta AI",
        "parameter_count_exact": 3_212_749_824,
        "parameter_count_display": "3.21 Billion (3.2B)",
        "is_generative": True,
        "model_type": "Causal Language Model (Transformer decoder)",
        "context_length_verified": 131072,
        "context_length_display": "131,072 tokens (128k)",
        "suitability_local_execution": "Suitable for Consumer Hardware (Requires 8 GB+ RAM if quantized)",
        "recommended_hardware": "NVIDIA GPU with 6-8 GB VRAM (for FP16) OR 8-16 GB system RAM (with INT4 quantization / Ollama)",
        "huggingface_url": "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
        "primary_use_cases": "Conversational assistant, local code completion, retrieval-augmented generation (RAG)",
        "license": "Llama 3.2 Community License (Permissive with attribution)"
    },
    "large": {
        "tier": "Large (~70 Billion Parameters)",
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "creator": "Meta AI",
        "parameter_count_exact": 70_553_698_304,
        "parameter_count_display": "70.6 Billion (70B)",
        "is_generative": True,
        "model_type": "Causal Language Model (Transformer decoder)",
        "context_length_verified": 131072,
        "context_length_display": "131,072 tokens (128k)",
        "suitability_local_execution": "NOT Suitable for Consumer Laptops (Data center / multi-GPU required)",
        "recommended_hardware": "Multi-GPU workstation (e.g., 2x NVIDIA A100 80GB for FP16, or 2x RTX 4090 with INT4 quantization). ~40-150 GB VRAM.",
        "huggingface_url": "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct",
        "primary_use_cases": "Complex mathematical reasoning, full-stack coding, enterprise synthetic data generation",
        "license": "Llama 3.1 Community License (Permissive with attribution)"
    }
}


def calculate_weight_memory_gb(param_count: int, bits_per_param: int) -> float:
    """
    Calculate theoretical model weight memory in Gigabytes (GB).
    
    Formula:
        Memory (Bytes) = Parameters * (bits / 8)
        Memory (GB)    = Memory (Bytes) / (1024^3)
    
    Note:
        This is a theoretical lower bound for static weights. Real runtime memory
        also includes the KV cache, activation buffers, CUDA runtime overhead,
        and operating system memory.
    """
    bytes_total = param_count * (bits_per_param / 8.0)
    gb_total = bytes_total / (1024 ** 3)
    return round(gb_total, 2)


def verify_model_live(model_id: str) -> Dict[str, Any]:
    """
    Attempts to fetch live metadata from the official Hugging Face Hub REST API.
    Handles network absence gracefully without crashing.
    """
    if not REQUESTS_AVAILABLE:
        return {"status": "unavailable", "reason": "requests library not installed"}
    
    api_url = f"https://huggingface.co/api/models/{model_id}"
    try:
        response = requests.get(api_url, timeout=4)
        if response.status_code == 200:
            data = response.json()
            return {
                "status": "online",
                "sha": data.get("sha", "N/A")[:10],
                "downloads": data.get("downloads", "N/A"),
                "likes": data.get("likes", "N/A"),
                "pipeline_tag": data.get("pipeline_tag", "N/A")
            }
        else:
            return {"status": "http_error", "code": response.status_code}
    except Exception as exc:
        return {"status": "offline", "reason": str(exc)}


def display_model_card(tier_key: str, model_info: Dict[str, Any]) -> None:
    """Prints a structured, student-level breakdown for a model."""
    print("=" * 80)
    print(f"TIER: {model_info['tier']}")
    print("=" * 80)
    print(f"Model ID            : {model_info['model_id']}")
    print(f"Creator/Org         : {model_info['creator']} [Verified]")
    print(f"Parameter Count     : {model_info['parameter_count_display']} [Verified: config.json]")
    print(f"Generative Model?   : {'Yes (Generates Text/Tokens)' if model_info['is_generative'] else 'No'} [Verified: CausalLM]")
    print(f"Architecture Type   : {model_info['model_type']} [Verified]")
    print(f"Context Window      : {model_info['context_length_display']} [Verified: max_position_embeddings]")
    print(f"Local Execution     : {model_info['suitability_local_execution']} [Assessment]")
    print(f"Recommended HW      : {model_info['recommended_hardware']} [Deployment Estimate]")
    print(f"Primary Use Cases   : {model_info['primary_use_cases']}")
    print(f"Model Card URL      : {model_info['huggingface_url']}")
    print(f"License             : {model_info['license']} [Verified]")

    # Memory calculations across precisions
    fp32_gb = calculate_weight_memory_gb(model_info["parameter_count_exact"], 32)
    fp16_gb = calculate_weight_memory_gb(model_info["parameter_count_exact"], 16)
    int8_gb = calculate_weight_memory_gb(model_info["parameter_count_exact"], 8)
    int4_gb = calculate_weight_memory_gb(model_info["parameter_count_exact"], 4)

    print("\n--- Theoretical Weight Memory Calculations (Static Weights Lower Bound) ---")
    print(f"  * FP32 (32-bit float, 4 bytes/param) : ~{fp32_gb} GB  [Theoretical Estimate]")
    print(f"  * FP16 (16-bit float, 2 bytes/param) : ~{fp16_gb} GB  [Theoretical Estimate - Standard GPU]")
    print(f"  * INT8 (8-bit integer, 1 byte/param) : ~{int8_gb} GB  [Theoretical Estimate]")
    print(f"  * INT4 (4-bit integer, 0.5 byte/param): ~{int4_gb} GB  [Theoretical Estimate - GGUF / Ollama]")
    print("  (Note: Real runtime requires additional RAM/VRAM for KV cache, activations, and OS overhead)")

    # Live API check
    live_result = verify_model_live(model_info["model_id"])
    print("\n--- Hugging Face API Verification ---")
    if live_result.get("status") == "online":
        print(f"  Status       : Verified reachable via Hugging Face REST API")
        print(f"  Pipeline Tag : {live_result.get('pipeline_tag')}")
        print(f"  All-time D/L : {live_result.get('downloads'):,}")
        print(f"  Likes        : {live_result.get('likes'):,}")
    else:
        print(f"  Status       : Offline / Cached Metadata Used ({live_result.get('status')})")
    print()


def print_theoretical_discussion() -> None:
    """Explains hardware factors influencing local execution."""
    print("=" * 80)
    print("THEORETICAL ANALYSIS & HARDWARE INFLUENCES")
    print("=" * 80)
    print("""
1. PARAMETER COUNT:
   - Each parameter represents a trained weight (floating-point number).
   - Higher parameter count allows richer representations and broader reasoning,
     but increases required memory linearly.

2. PRECISION:
   - FP16 (16-bit) requires 2 bytes per parameter.
   - INT4 quantization compresses each weight to 4 bits (0.5 byte), saving ~75% memory.
   - However, quantization introduces minor quantization noise, which may affect
     nuanced reasoning or mathematical precision.

3. CONTEXT LENGTH & THE KV CACHE:
   - When generating tokens, the model caches key and value projections for all previous
     tokens in the prompt to avoid recomputing self-attention at each step.
   - At long context lengths (e.g., 128k tokens for Llama-3.2), the KV cache can consume
     several gigabytes of extra VRAM/RAM beyond the model weights.

4. SYSTEM RAM VS. GPU VRAM:
   - GPUs feature high-bandwidth memory (VRAM), often delivering 500 to 1,000+ GB/s.
   - Standard DDR4/DDR5 system RAM provides only 40 to 90 GB/s.
   - Because LLM token generation is strictly memory-bandwidth bound (reading all weights
     for every generated token), running models on CPU RAM is typically 10x to 50x slower
     than running on a dedicated GPU.
""")


def main() -> None:
    """Main entry point to execute Activity 1."""
    print("=" * 80)
    print("LAB 1: ACTIVITY 1 - EXPLORE LLM MODELS ON HUGGING FACE")
    print("=" * 80)
    print("Investigating 3 representative model tiers: Small, Medium, and Large.\n")

    for tier_key in ["small", "medium", "large"]:
        display_model_card(tier_key, MODELS_CATALOG[tier_key])

    print_theoretical_discussion()
    print("Activity 1 execution complete.")


if __name__ == "__main__":
    main()
