"""
Lab 1 — Activity 4: Can My Computer Run an LLM?
Author: Shubranil Pandit (SCFP125024)
Course: MCA 3rd Semester — Generative AI Lab

WHAT:
    This activity benchmarks local hardware capabilities (RAM, CPU, and GPU) on Windows
    and applies theoretical memory estimation formulas across precisions (FP16 vs INT4).
    It rigorously analyzes whether a computer—specifically an 8 GB RAM laptop—can safely
    run 4-billion and 7-billion parameter language models.

WHY:
    Deploying an LLM locally without calculating memory requirements leads to Out-Of-Memory (OOM)
    crashes, severe OS paging/thrashing, or system freezing. An AI engineer must be able
    to estimate static weight footprint, account for dynamic runtime memory (KV cache,
    activations, runtime context), and select appropriate quantization levels.

HOW:
    - Queries physical RAM, available RAM, and GPU hardware using `psutil` and Windows WMI/CIM.
    - Implements the foundational estimation formula:
        Theoretical Weight Memory (Bytes) = Parameter Count * (Bits per Parameter / 8)
    - Compares 4B FP16, 4B INT4, and 7B INT4 model requirements.
    - Analyzes dynamic memory consumers beyond static weights (KV cache, OS, activations).
    - Solves the assignment's 8 GB laptop dilemma with realistic engineering calculations.

OBSERVATION:
    - This computer has 16.0 GB total physical RAM (~6.6 GB currently free) and integrated
      Intel UHD Graphics without dedicated high-speed VRAM.
    - 4B FP16 requires ~7.45 GB for weights alone, making it impossible on an 8 GB laptop.
    - 7B INT4 requires ~3.26 GB for weights, but total system usage exceeds 7.5 GB when
      accounting for Windows OS and KV cache, creating severe memory pressure on an 8 GB laptop.
    - 4B INT4 requires ~1.86 GB for weights (~2.2 GB with runtime overhead). While it is the
      most viable option among the three, it is NOT universally guaranteed to work smoothly
      on every 8 GB machine unless background applications are restricted and context length is capped.

CONCLUSION:
    Actual memory consumption significantly exceeds static model weights. On an 8 GB laptop,
    operating system overhead leaves only 4-5 GB of usable memory. A 4B INT4 quantized model
    is the most realistic candidate, but execution stability depends on background memory
    pressure, integrated GPU shared memory, and context window limits.
"""

import sys
import os
import subprocess
from typing import Dict, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def detect_system_hardware() -> Dict[str, Any]:
    """
    Detects local Windows RAM and GPU hardware.
    Uses psutil for memory and Windows CIM/PowerShell for GPU detection.
    """
    hardware_info = {
        "os_name": sys.platform,
        "total_ram_gb": 0.0,
        "available_ram_gb": 0.0,
        "used_ram_percent": 0.0,
        "gpu_name": "No discrete GPU detected (or integrated graphics only)",
        "has_nvidia_cuda": False
    }

    # 1. Detect RAM using psutil
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        hardware_info["total_ram_gb"] = round(mem.total / (1024 ** 3), 2)
        hardware_info["available_ram_gb"] = round(mem.available / (1024 ** 3), 2)
        hardware_info["used_ram_percent"] = mem.percent

    # 2. Query Windows Video Controller (GPU) via PowerShell CIM
    try:
        cmd = ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        gpu_output = result.stdout.strip()
        if gpu_output:
            # Can return multiple lines if both integrated and discrete GPUs exist
            gpus = [g.strip() for g in gpu_output.splitlines() if g.strip()]
            hardware_info["gpu_name"] = " / ".join(gpus)
            if any("nvidia" in g.lower() for g in gpus):
                hardware_info["has_nvidia_cuda"] = True
    except Exception:
        pass

    return hardware_info


def estimate_weight_memory(param_count_billions: float, bits_per_param: int) -> Dict[str, float]:
    """
    Estimates theoretical model-weight memory requirements.
    
    Formula:
        Weight Memory (Bytes) = Parameter Count * (bits_per_param / 8)
        Weight Memory (GB)    = Weight Memory (Bytes) / (1024^3)
    
    Also provides a realistic deployment estimate including ~15% GGUF/quantization metadata
    overhead (scales, zero-points, block offsets).
    """
    param_count = param_count_billions * 1e9
    bytes_theoretical = param_count * (bits_per_param / 8.0)
    gb_theoretical = bytes_theoretical / (1024 ** 3)
    
    # Quantized formats (GGUF, AWQ, GPTQ) incur ~10-15% overhead for scales and metadata
    real_overhead_multiplier = 1.15 if bits_per_param <= 8 else 1.05
    gb_realistic = gb_theoretical * real_overhead_multiplier

    return {
        "theoretical_gb": round(gb_theoretical, 2),
        "realistic_gb": round(gb_realistic, 2)
    }


def display_estimation_matrix() -> None:
    """
    Displays the three specific scenarios requested by the assignment:
      1. 4B model using FP16 (16 bits)
      2. 4B model using INT4 (4 bits)
      3. 7B model using INT4 (4 bits)
    """
    scenarios = [
        ("4-Billion Parameters", 4.0, 16, "FP16 (Half Precision)"),
        ("4-Billion Parameters", 4.0, 4,  "INT4 (4-bit Quantization)"),
        ("7-Billion Parameters", 7.0, 4,  "INT4 (4-bit Quantization)"),
    ]

    print("=" * 80)
    print("THEORETICAL WEIGHT MEMORY ESTIMATION (Basic Formula: P * bits / 8)")
    print("=" * 80)
    print(f"{'Model Size':<22} | {'Precision':<24} | {'Theoretical':<14} | {'Realistic (w/ Overhead)'}")
    print("-" * 80)

    for name, params, bits, prec_label in scenarios:
        res = estimate_weight_memory(params, bits)
        theo_str = f"~{res['theoretical_gb']:.2f} GB"
        real_str = f"~{res['realistic_gb']:.2f} GB (GGUF/weights)"
        print(f"{name:<22} | {prec_label:<24} | {theo_str:<14} | {real_str}")

    print("-" * 80)
    print("Note: 1 GB = 1024^3 bytes (GiB). Estimates represent static weight storage only.")


def analyze_8gb_laptop_scenario() -> None:
    """
    Answers the primary analytical question from Lab 1 Activity 4:
    'Your laptop has 8 GB RAM — 4B FP16, 4B INT4, or 7B INT4? Why?'
    """
    print("\n" + "=" * 80)
    print("CASE STUDY: EVALUATING AN 8 GB RAM LAPTOP WITHOUT A GPU")
    print("=" * 80)

    print("""
1. THE 8 GB RAM REALITY CHECK:
   - Total Physical RAM: 8.0 GB
   - Windows Operating System & Background Services: ~3.0 - 3.8 GB
   - Truly Available RAM for LLM: ~4.2 - 5.0 GB maximum!

2. CANDIDATE MODEL EVALUATION:

   [A] 4-Billion Parameters @ FP16 (16-bit):
       - Weight Memory: ~7.45 GB
       - Total System Memory Needed: 7.45 GB (weights) + 3.5 GB (OS) = ~11.0 GB
       - VERDICT: FAILS COMPLETELY (Immediate Out-Of-Memory crash or severe disk thrashing).

   [B] 7-Billion Parameters @ INT4 (4-bit):
       - Weight Memory: ~3.26 GB (theoretical), ~3.75 GB (with GGUF metadata)
       - Runtime KV Cache (4k context) + Activations: ~0.8 GB
       - Total System Memory Needed: ~3.75 + 0.8 + 3.5 (OS) = ~8.05 GB
       - VERDICT: EXTREMELY RISKY / UNSTABLE. It pushes the machine to 99% RAM usage.
         Opening a single browser tab or generating a longer response will force
         Windows to page memory to the hard drive, freezing the laptop.

   [C] 4-Billion Parameters @ INT4 (4-bit):
       - Weight Memory: ~1.86 GB (theoretical), ~2.15 GB (with GGUF metadata)
       - Runtime KV Cache (2k context) + Activations: ~0.4 - 0.6 GB
       - Baseline System Memory Needed: ~2.15 + 0.5 + 3.5 (OS) = ~6.15 GB
       - Potential Headroom: ~1.85 GB under clean boot conditions.
       - VERDICT: THE MOST FEASIBLE CANDIDATE AMONG THE THREE.
         However, it is NOT UNIVERSALLY GUARANTEED to run without hiccups.
         Real-world caveats:
         * If the user runs background apps (e.g. Chrome, VS Code, Discord, or Teams),
           free RAM can drop below 2 GB before the LLM even launches.
         * Integrated graphics (Intel UHD / Iris Xe / AMD Radeon) allocate VRAM
           dynamically from the same system RAM (often 512 MB to 2 GB).
         * Extended context lengths (> 2048 tokens) will rapidly expand the KV cache.
         * Recommended setup on an 8 GB machine: close background applications, use
           small context windows (e.g. 2048 tokens), and consider a 1B-3B model if
           multitasking is required.
""")


def print_theoretical_discussion() -> None:
    """Explains memory components beyond weights and quantization trade-offs."""
    print("=" * 80)
    print("THEORETICAL ANALYSIS: BEYOND MODEL WEIGHTS & QUANTIZATION TRADE-OFFS")
    print("=" * 80)
    print("""
Q1: What else besides model weights uses memory during inference?
A1: Model weights are only the static baseline. Real inference requires:
    1. Operating System & Drivers: Windows typically consumes 3-4 GB just staying alive.
    2. KV (Key-Value) Cache: Stores attention keys and values for all past tokens in the
       session. Memory grows with context length:
       Formula: KV_Cache = 2 * n_layers * n_heads * d_head * seq_len * precision_bytes
       At 8k context, this can add 1 to 2 GB of memory!
    3. Activation Memory: Temporary buffers used to store intermediate matrix multiplication
       results during forward propagation.
    4. Runtime Engine Overhead: Memory allocated by Ollama / llama.cpp for thread pools,
       scratchpads, and token buffers (~200-400 MB).

Q2: Would quantization always be the safer choice? Why or why not?
A2: Quantization is usually the safer choice for MEMORY, but NOT ALWAYS universally:
    Pros:
      - Slashes memory footprint by up to 75% (FP16 -> INT4).
      - Greatly accelerates CPU inference because CPU LLM execution is memory-bandwidth bound.
    Cons & Limitations:
      - Loss of Precision / Perplexity Degradation: Compressing 16-bit continuous values into
        discrete 4-bit bins introduces quantization noise. This can slightly impair delicate
        mathematical reasoning, code generation, or nuanced multilingual capabilities.
      - Compute Overhead: On GPUs with FP16 tensor cores but without native INT4 acceleration,
        the engine must dequantize weights back to FP16 in registers before multiplication.
      - Severe Degradation Below 3-Bits: 4-bit is the sweet spot. 2-bit or 1.5-bit quantization
        leads to severe model degradation ('model lobotomy').
""")


def main() -> None:
    """Main execution point for Activity 4."""
    print("=" * 80)
    print("LAB 1: ACTIVITY 4 - CAN MY COMPUTER RUN AN LLM?")
    print("=" * 80)

    # 1. Detect and print current machine specifications
    hw = detect_system_hardware()
    print("CURRENT LOCAL SYSTEM SPECIFICATIONS:")
    print(f"  Operating System  : {hw['os_name'].capitalize()} (Windows)")
    print(f"  Total System RAM  : {hw['total_ram_gb']} GB")
    print(f"  Available RAM     : {hw['available_ram_gb']} GB ({100 - hw['used_ram_percent']:.1f}% free)")
    print(f"  Detected Graphics : {hw['gpu_name']}")
    print(f"  Discrete CUDA GPU : {'Yes' if hw['has_nvidia_cuda'] else 'No (Using CPU / Shared Memory)'}\n")

    # 2. Display mathematical estimation matrix
    display_estimation_matrix()

    # 3. Solve 8 GB laptop dilemma
    analyze_8gb_laptop_scenario()

    # 4. Theoretical discussion and viva prep
    print_theoretical_discussion()

    print("Activity 4 execution complete.")


if __name__ == "__main__":
    main()
