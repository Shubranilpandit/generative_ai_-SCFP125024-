"""
Lab 1 — Activity 7: Compare Local vs. Colab vs. API
Author: Shubranil Pandit (SCFP125024)
Course: MCA 3rd Semester — Generative AI Lab

WHAT:
    This activity performs an analytical comparison of the three primary paradigm
    approaches for executing Large Language Models:
      1. Local On-Device Inference (e.g., Ollama / llama.cpp)
      2. Cloud-Hosted GPU Notebooks (e.g., Google Colab T4/A100)
      3. Hosted API / Inference Providers (e.g., Groq / OpenAI / Anthropic)
    It evaluates them across 8 practical engineering dimensions, presents a structured
    comparison matrix, and provides a reasoned recommendation for an 8 GB non-GPU laptop.

WHY:
    In real-world GenAI engineering, there is no single "best" execution architecture.
    Choosing the right deployment strategy requires balancing latency, hardware costs,
    scalability, setup complexity, internet dependency, and data privacy regulations (GDPR/HIPAA).

HOW:
    - Encapsulates comparison criteria into structured data dictionaries.
    - Generates a readable, formatted comparison table across all 8 dimensions.
    - Systematically analyzes the assignment's core questions (8 GB laptop dilemma,
      privacy extremes, scalability ceilings, and decision criteria).
    - Highlights security and privacy nuances honestly without making blanket claims.

OBSERVATION:
    - An 8 GB RAM non-GPU laptop struggles with models larger than 3B-4B locally (~3-5 tokens/sec).
    - Google Colab provides free GPU VRAM (~15 GB T4) for interactive prototyping, but is
      constrained by idle timeouts and session resets.
    - Hosted APIs offer instantaneous responses (~60-200+ tokens/sec) and require zero local RAM,
      making them the most practical choice for day-to-day experimentation on low-spec laptops.

CONCLUSION:
    For an 8 GB RAM laptop with no GPU, a hybrid strategy is optimal: use a Hosted API for fast
    iteration and general tasks, Google Colab when PyTorch CUDA access is required, and a small
    local quantized model (1B to 3B INT4 via Ollama) when working offline or with private data.
"""

import sys
from typing import List, Dict, Any


def get_comparison_matrix() -> List[Dict[str, str]]:
    """Returns structured data comparing the three execution paradigms."""
    return [
        {
            "dimension": "Hardware Demands",
            "local": "High (Needs 8-16 GB+ RAM & multi-core CPU/GPU)",
            "colab": "Zero local (Uses Google Cloud Tesla T4/V100)",
            "api": "Zero local (Any basic PC, phone, or tablet)"
        },
        {
            "dimension": "Inference Latency",
            "local": "Slow on CPU (~3-6 tok/s); Fast on dedicated GPU",
            "colab": "Fast (~30-60 tok/s on T4 GPU)",
            "api": "Ultra Fast (~60-250+ tok/s on LPUs/H100s)"
        },
        {
            "dimension": "Setup Difficulty",
            "local": "Moderate (Install Ollama, pull GGUF models)",
            "colab": "Low-Moderate (Google account, notebook cells)",
            "api": "Very Low (Sign up, obtain key, pip install sdk)"
        },
        {
            "dimension": "Cost Model",
            "local": "Free (Zero marginal cost after hardware purchase)",
            "colab": "Free tier available; Compute units for Pro",
            "api": "Pay-per-token (e.g. $0.05 - $0.80 / 1M tokens)"
        },
        {
            "dimension": "Internet Dependency",
            "local": "Zero (100% offline once model is downloaded)",
            "colab": "Strict (Continuous connection required)",
            "api": "Strict (Fails immediately if offline)"
        },
        {
            "dimension": "Privacy & Data Control",
            "local": "Highest (Data never leaves local device memory)",
            "colab": "Moderate (Data stored in Google temporary VM)",
            "api": "Lowest (Prompts traverse internet & cloud infra)"
        },
        {
            "dimension": "Scalability (Multi-User)",
            "local": "None (Limited to single local user / laptop)",
            "colab": "None (Interactive single-notebook environment)",
            "api": "Infinite (Autoscaling enterprise cloud clusters)"
        },
        {
            "dimension": "Max Model Capability",
            "local": "Limited to 1B - 7B (on consumer laptops)",
            "colab": "Limited to ~7B - 14B (within 15 GB VRAM)",
            "api": "Huge models available (70B, 405B, GPT-4, Claude)"
        }
    ]


def display_comparison_table() -> None:
    """Prints a formatted ASCII comparison table."""
    data = get_comparison_matrix()
    print("=" * 105)
    print("COMPREHENSIVE PARADIGM COMPARISON: LOCAL (OLLAMA) vs. GOOGLE COLAB vs. HOSTED API")
    print("=" * 105)
    print(f"{'Evaluation Dimension':<22} | {'Local (Ollama)':<26} | {'Google Colab (GPU)':<26} | {'Hosted API (Groq/OpenAI)'}")
    print("-" * 105)

    for row in data:
        print(f"{row['dimension']:<22} | {row['local']:<26} | {row['colab']:<26} | {row['api']}")

    print("-" * 105)


def analyze_8gb_laptop_decision() -> None:
    """Answers: If you had an 8 GB RAM laptop with no GPU, which approach would you pick?"""
    print("\n" + "=" * 105)
    print("SCENARIO ANALYSIS: 8 GB RAM LAPTOP WITHOUT A DISCRETE GPU")
    print("=" * 105)
    print("""
If I had an 8 GB RAM laptop with no dedicated GPU, my reasoned selection would be:

1. PRIMARY CHOICE FOR DAILY WORK & LAB ASSIGNMENTS: HOSTED API (e.g. Groq)
   Why?
   - Zero Hardware Burden: The laptop acts merely as a lightweight client sending JSON
     over HTTPS. The 8 GB RAM is completely preserved for the OS, IDE, and web browser.
   - High Speed & Superior Intelligence: Instead of being restricted to a tiny 1B-3B
     quantized model generating at ~4 tokens/second on CPU, an API provides access to
     massive 20B-70B models at blazing speeds (60 to 200+ tokens/second).
   - Zero Battery & Fan Strain: Generating tokens locally on CPU throttles laptop fans
     and drains battery rapidly; API calls consume negligible power.

2. SECONDARY CHOICE FOR CUDA / PYTORCH LEARNING: GOOGLE COLAB GPU
   Why?
   - When assignments require inspecting PyTorch tensors, examining attention weights,
     fine-tuning with LoRA, or testing CUDA operations (like Activity 3), Colab provides
     a free Tesla T4 GPU with 15 GB of high-speed VRAM that the physical laptop lacks.

3. FALLBACK FOR OFFLINE / CONFIDENTIAL DATA: OLLAMA WITH A 1.5B - 3B INT4 MODEL
   Why?
   - If Wi-Fi is disconnected, or if analyzing proprietary/confidential data that must
     never leave the device, local Ollama is the only viable option. On an 8 GB laptop,
     running a 1.5B or 3B INT4 model (such as Qwen2.5-1.5B or 3B) leaves sufficient RAM headroom
     to prevent system lockups.
   - Real-World Caveats on 8 GB RAM:
     * Background memory discipline is essential: browsers with many tabs, Discord, or Teams
       must be closed to prevent disk paging/thrashing.
     * Integrated graphics (Intel UHD) dynamically reserve system RAM for display buffers.
     * Keep context length capped (<= 2048 tokens) to prevent the KV cache from expanding.
""")


def print_theoretical_answers() -> None:
    """Answers the remaining theoretical questions from Lab 1 Activity 7."""
    print("=" * 105)
    print("THEORETICAL ANALYSIS & ASSIGNMENT QUESTIONS")
    print("=" * 105)
    print("""
Q1: Which approach is most private? Which is most scalable?
A1:
    - MOST PRIVATE: Local Execution (Ollama).
      All model weights, prompt tokens, and KV cache remain strictly inside your machine's
      RAM/VRAM. No external network requests are initiated.
      *Caveat:* Local privacy assumes the machine itself is secure, free of malware, and
      not running unverified third-party telemetry integrations.
    - MOST SCALABLE: Hosted APIs.
      Cloud providers manage massive data centers with load balancers, auto-scaling worker
      pools, and global CDN caching. A hosted API can scale from 1 request/day to 100,000
      concurrent requests without the developer purchasing a single server. Local laptops
      and Google Colab notebooks cannot scale to serve concurrent multi-user workloads.

Q2: When would you choose local over cloud, or the reverse?
A2:
    CHOOSE LOCAL (OLLAMA / GGUF) WHEN:
      1. Privacy & Compliance: Regulated industries (healthcare, legal, banking, defence)
         with strict data residency rules (GDPR, HIPAA) prohibiting data transfer to 3rd parties.
      2. Offline Environments: Field robotics, maritime vessels, aircraft, remote areas, or
         air-gapped secure development labs.
      3. Zero Marginal Cost: When processing millions of internal background tasks where
         accumulating token API bills would be cost-prohibitive.

    CHOOSE CLOUD (API / COLAB) WHEN:
      1. Advanced Reasoning: When the task requires the knowledge and coding capability of
         large frontier models (70B to 405B parameters) that no laptop can fit.
      2. Resource-Constrained Hardware: When working on thin laptops, Chromebooks, or mobile devices.
      3. Speed & User Experience: When an interactive, sub-second typing experience is required.
      4. Rapid Prototyping: When you want to build and deploy applications immediately without
         managing drivers, CUDA dependencies, model downloads, or server cooling.
""")


def main() -> None:
    """Main execution point for Activity 7."""
    print("=" * 105)
    print("LAB 1: ACTIVITY 7 - COMPARE LOCAL VS. COLAB VS. API")
    print("=" * 105)

    display_comparison_table()
    analyze_8gb_laptop_decision()
    print_theoretical_answers()
    print("Activity 7 execution complete.")


if __name__ == "__main__":
    main()
