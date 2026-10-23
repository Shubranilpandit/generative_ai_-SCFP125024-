"""
Lab 1 — Activity 3: Run an LLM Using Google Colab GPU
Author: Shubranil Pandit (SCFP125024)
Course: MCA 3rd Semester — Generative AI Lab

WHAT:
    This activity provides a complete, runnable script designed to execute a small
    generative language model (SmolLM2-135M or Qwen2.5-0.5B) on an NVIDIA GPU in Google Colab.
    It verifies CUDA acceleration, profiles VRAM usage before and after loading, measures
    inference latency, and studies the latency impact of increasing output token length.

WHY:
    Large Language Models rely heavily on parallel matrix operations and high memory bandwidth.
    While consumer laptops often lack dedicated GPUs or high-speed VRAM, cloud environments
    like Google Colab provide free or low-cost access to data center GPUs (such as the NVIDIA T4).
    Understanding how to inspect GPU devices, measure memory allocation, and observe autoregressive
    token latency is a core skill for LLM practitioners.

HOW:
    - Detects whether execution is occurring in Google Colab or a local environment.
    - Uses PyTorch (`torch.cuda.is_available()`) to query GPU availability.
    - Inspects GPU properties: device name, total VRAM, allocated memory, and reserved memory.
    - Loads a small instruction-tuned model (`HuggingFaceTB/SmolLM2-135M-Instruct` or `gpt2`).
    - Benchmarks text generation at two different output lengths (`max_new_tokens=25` vs `100`).
    - Gracefully handles local execution on a non-GPU Windows machine, providing full
      instructions for copying and executing inside Colab.

OBSERVATION:
    - On Colab T4 GPU: Model loading takes ~2-4s; generation runs at ~40-80 tokens/sec.
      VRAM consumption is ~300-500 MB for a 135M model.
    - Latency scales linearly with output token length because autoregressive generation
      produces one token per forward pass (each requiring reading the full model weights).
    - On Local Windows without GPU: CUDA is unavailable. A local process cannot magically
      use Google's cloud GPU without running directly inside Colab's hosted runtime.

CONCLUSION:
    GPUs accelerate LLM inference by an order of magnitude compared to CPUs due to VRAM bandwidth.
    For students on laptops without dedicated NVIDIA GPUs, Google Colab provides a free,
    accessible environment to experiment with CUDA-accelerated models.
"""

import sys
import time
import os
from typing import Dict, Any


def is_running_in_colab() -> bool:
    """Detects whether the script is currently running inside Google Colab."""
    return "google.colab" in sys.modules or os.path.exists("/content")


def check_gpu_environment() -> Dict[str, Any]:
    """
    Inspects PyTorch CUDA availability and queries hardware specifications.
    Honest reporting: never claims GPU is active unless PyTorch confirms it.
    """
    env_info = {
        "colab": is_running_in_colab(),
        "torch_installed": False,
        "cuda_available": False,
        "device_name": "None",
        "total_vram_gb": 0.0,
        "allocated_vram_mb": 0.0,
        "reserved_vram_mb": 0.0,
    }

    try:
        import torch
        env_info["torch_installed"] = True
        env_info["torch_version"] = torch.__version__

        if torch.cuda.is_available():
            env_info["cuda_available"] = True
            env_info["device_name"] = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            env_info["total_vram_gb"] = round(total_bytes / (1024 ** 3), 2)
            env_info["allocated_vram_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)
            env_info["reserved_vram_mb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2)
    except ImportError:
        pass

    return env_info


def display_colab_setup_guide() -> None:
    """Prints step-by-step instructions for running this file in Google Colab."""
    print("=" * 80)
    print("HOW TO RUN THIS ACTIVITY IN GOOGLE COLAB WITH A FREE GPU")
    print("=" * 80)
    print("""
Step 1: Open Google Colab in your browser:
        https://colab.research.google.com

Step 2: Enable GPU Hardware Acceleration:
        - In the Colab menu, click: 'Runtime' -> 'Change runtime type'
        - Under 'Hardware accelerator', select 'T4 GPU'
        - Click 'Save'

Step 3: In the first Colab cell, install dependencies:
        !pip install --quiet torch transformers accelerate

Step 4: Copy the contents of this file (activity3.py) into a code cell or upload it:
        !python activity3.py

Note on Local Execution:
    A Python process running on your local Windows PC CANNOT automatically use
    Google's cloud GPU. Google's T4 GPUs reside in Google's remote data centers.
    To use Google's GPU, the code must execute inside the Colab browser runtime.
""")


def run_model_inference_benchmark(env_info: Dict[str, Any]) -> None:
    """
    Loads a lightweight model and benchmarks inference at different token lengths.
    Uses GPU if available, or CPU if PyTorch is installed locally.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as err:
        print(f"[!] Cannot run inference benchmark: {err}")
        print("    In Colab, run: !pip install torch transformers accelerate")
        return

    device = "cuda" if env_info["cuda_available"] else "cpu"
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

    print("=" * 80)
    print(f"LOADING MODEL FOR INFERENCE BENCHMARK ({model_id})")
    print(f"Target Device: {device.upper()}")
    print("=" * 80)

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Measure Model Loading Time
    t_start = time.time()
    print(f"Loading model weights into {device.upper()} memory...")
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.to(device)

    load_time = time.time() - t_start
    print(f"Model loaded successfully in {load_time:.2f} seconds.")

    # 3. Memory profiling if on GPU
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"VRAM Allocated by Weights: {allocated:.1f} MB")
        print(f"VRAM Reserved by PyTorch: {reserved:.1f} MB")

    # 4. Benchmarking generation at different output lengths
    prompt = "Explain in one sentence why GPU memory bandwidth matters for LLMs."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("\n" + "-" * 80)
    print(f"Prompt: \"{prompt}\"")
    print("-" * 80)

    for max_tokens in [25, 75]:
        print(f"\n--- Benchmark: max_new_tokens = {max_tokens} ---")
        t_gen_start = time.time()
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy decoding for consistent timing
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - t_gen_start
        generated_count = len(output_tokens[0]) - len(inputs["input_ids"][0])
        throughput = generated_count / gen_time if gen_time > 0 else 0.0

        response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(f"Generation Time : {gen_time:.3f} seconds")
        print(f"Tokens Generated: {generated_count} tokens")
        print(f"Speed           : {throughput:.1f} tokens/second")
        print(f"Output Preview  : \"{response_text.strip()[:140]}...\"")

        if device == "cuda":
            curr_vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
            print(f"Active VRAM     : {curr_vram:.1f} MB (includes KV cache)")


def print_theoretical_discussion() -> None:
    """Answers the four core questions from Lab 1 Activity 3."""
    print("\n" + "=" * 80)
    print("ASSIGNMENT QUESTIONS & EXPERIMENTAL ANALYSIS")
    print("=" * 80)
    print("""
Q1: Is the GPU actually being used?
A1: Yes, in Google Colab when CUDA is active. PyTorch confirms this via
    'torch.cuda.is_available() == True' and assigns tensors to 'cuda:0' (e.g. Tesla T4).
    You can also verify GPU utilization in Colab by running '!nvidia-smi' or viewing
    the Colab Resource Monitor tab.
    On a local non-GPU Windows laptop, 'torch.cuda.is_available()' returns False,
    meaning execution falls back to CPU.

Q2: How much memory does the model consume?
A2: For a 135M model (SmolLM2-135M) in FP16 precision:
    - Model weights consume ~270 MB VRAM.
    - PyTorch CUDA context and driver overhead add ~200-300 MB.
    - Total baseline VRAM is ~500 MB.
    For a 3B model (e.g., Llama-3.2-3B) in FP16:
    - Model weights alone consume ~6.4 GB VRAM.

Q3: What happens if you increase the output length?
A3: Two distinct things happen:
    1. Latency increases linearly: Because LLMs are autoregressive, generating 100 tokens
       requires 100 sequential forward passes. Each pass must read all model weights
       from memory.
    2. Memory consumption grows: The Key-Value (KV) cache grows with each new token
       generated to maintain historical attention, consuming additional VRAM.

Q4: Would this run on a laptop without a GPU?
A4: - A tiny 135M or 500M model CAN run on a standard CPU laptop with acceptable speed
      (~10-25 tokens/sec).
    - A 3B or 7B model can run on CPU RAM only if quantized (e.g., INT4 GGUF via Ollama),
      but generation speed drops to ~3-8 tokens/sec on CPU compared to ~40+ tokens/sec
      on a dedicated GPU, due to the limited memory bandwidth of standard DDR4/DDR5 RAM.
""")


def main() -> None:
    """Main execution point for Activity 3."""
    print("=" * 80)
    print("LAB 1: ACTIVITY 3 - RUN AN LLM USING GOOGLE COLAB GPU")
    print("=" * 80)

    env = check_gpu_environment()
    print(f"Runtime Environment : {'Google Colab' if env['colab'] else 'Local Machine (Windows)'}")
    print(f"PyTorch Installed   : {env['torch_installed']}")
    print(f"CUDA GPU Available  : {env['cuda_available']}")

    if env["cuda_available"]:
        print(f"GPU Device Name     : {env['device_name']}")
        print(f"Total GPU VRAM      : {env['total_vram_gb']} GB")
        print(f"Currently Allocated : {env['allocated_vram_mb']} MB")
        print(f"Currently Reserved  : {env['reserved_vram_mb']} MB")
        # Run live benchmark on GPU
        run_model_inference_benchmark(env)
    else:
        print("\n[!] Status: No active NVIDIA CUDA GPU detected in the current Python environment.")
        print("    If running locally on Windows: This is expected if your laptop uses integrated")
        print("    graphics (Intel UHD) rather than a discrete NVIDIA GPU.")
        print()
        display_colab_setup_guide()

    print_theoretical_discussion()
    print("Activity 3 execution complete.")


if __name__ == "__main__":
    main()
