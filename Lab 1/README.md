# Lab 1 — Introduction to LLMs: Practical Activities

**Student Name:** Shubranil Pandit  
**Roll / ID:** SCFP125024  
**Course:** MCA 3rd Semester — Generative AI Lab  
**Repository:** [generative_ai_-SCFP125024-](https://github.com/Shubranilpandit/generative_ai_-SCFP125024-.git)  
**Operating System:** Windows 11  

---

## 1. Objectives

The primary objective of this laboratory assignment is to develop an in-depth, hands-on understanding of Large Language Models (LLMs) from an engineering perspective:
- Exploring real open-source LLM architectures across model scales (Small, Medium, Large).
- Demystifying subword tokenization, token IDs, vocabulary limits, and detokenization.
- Running models using GPU acceleration in Google Colab and profiling VRAM / latency.
- Analyzing hardware bottlenecks (RAM, GPU VRAM, memory bandwidth, KV cache, and quantization).
- Running local open-source models with Ollama and comparing with remote API inference.
- Investigating generation parameters (Temperature sampling) and building conversational chatbots.

---

## 2. Directory Structure

```text
Lab 1/
├── Lab1_Introduction_to_LLMs.md   # Official lab assignment instructions
├── README.md                      # Comprehensive lab guide and documentation
├── requirements.txt               # Minimal project dependencies
├── .gitignore                     # Git ignore rules protecting credentials & artifacts
├── .env.example                   # Environment configuration placeholder template
├── activity1.py                   # Activity 1: Explore LLM Models on Hugging Face
├── activity2.py                   # Activity 2: Tokenization Practical
├── activity3.py                   # Activity 3: Run an LLM Using Google Colab GPU
├── activity4.py                   # Activity 4: Can My Computer Run an LLM?
├── activity5.py                   # Activity 5: Run an LLM Locally Using Ollama
├── activity6.py                   # Activity 6: Use an LLM Through an API Provider
├── activity7.py                   # Activity 7: Compare Local vs. Colab vs. API
├── activity8.py                   # Activity 8: Temperature Experiment
├── activity9.py                   # Activity 9: Build a Mini Chatbot
└── activity10.py                  # Activity 10: Customize the Chatbot (AI/ML Tutor)
```

---

## 3. Prerequisites & Environment Setup

### 3.1 Python Environment
This project runs on Python 3.10+ (tested on Python 3.14 / 3.11).

```powershell
# Navigate to the repository root
cd "C:\Shubranil\College\Notes\Sem-3\Lab Subject\GenAI\Lab Assignments\SCFP125024_MCA_3rd_Sem"

# If activating the existing virtual environment:
.\myenv\Scripts\activate

# Or creating a fresh virtual environment:
py -m venv .venv
.\.venv\Scripts\activate
```

### 3.2 Dependency Installation
Install the required dependencies using pip:

```powershell
pip install -r "Lab 1\requirements.txt"
```

### 3.3 Environment Configuration (`.env`)
Copy `.env.example` to `.env` if using API providers for upcoming activities:

```powershell
cp "Lab 1\.env.example" "Lab 1\.env"
```

> **Security Note:** Never commit `.env` or actual API keys to GitHub. `.env` is strictly excluded in `.gitignore`.

---

## 4. How to Run Activities (Phase 3: Activities 1 to 4)

### Activity 1 — Explore LLM Models on Hugging Face
Inspects three verified Hugging Face models across size tiers:
- **Small:** `HuggingFaceTB/SmolLM2-135M-Instruct` (~135M parameters, 8k context)
- **Medium:** `meta-llama/Llama-3.2-3B-Instruct` (~3.21B parameters, 128k context)
- **Large:** `meta-llama/Llama-3.1-70B-Instruct` (~70.6B parameters, 128k context)

```powershell
python "Lab 1\activity1.py"
```

### Activity 2 — Tokenization Practical
Demonstrates subword tokenization with Hugging Face's `AutoTokenizer` (GPT-2 BPE):
- Demonstrates why 1 word $\neq$ 1 token.
- Analyzes leading spaces, punctuation marks, compound words, and rare words.
- Explains why neural networks require discrete integer Token IDs.

```powershell
python "Lab 1\activity2.py"
```

### Activity 3 — Run an LLM Using Google Colab GPU
Tests PyTorch CUDA availability and profiles VRAM allocation and generation latency:
- Explains that local Windows processes cannot access Google Colab's cloud GPU directly.
- Provides step-by-step instructions to run the script inside Google Colab with a free T4 GPU.
- Measures how increasing `max_new_tokens` increases autoregressive latency linearly.

```powershell
python "Lab 1\activity3.py"
```

### Activity 4 — Can My Computer Run an LLM?
Profiles local Windows hardware (RAM & GPU) and calculates theoretical memory requirements:
- Computes weight memory using $P \times \text{bits} / 8$ for **4B FP16** (~7.45 GB), **4B INT4** (~1.86 GB), and **7B INT4** (~3.26 GB).
- Evaluates the **8 GB RAM laptop scenario**: Why 4B INT4 is the most realistic candidate among the three (with proper context limits and closing background apps), while 4B FP16 and 7B INT4 face severe memory exhaustion.
- Explains memory components beyond weights and the trade-offs of quantization.

```powershell
python "Lab 1\activity4.py"
```

### Activity 5 — Run an LLM Locally Using Ollama
Interacts with the local Ollama service (`http://localhost:11434`) running on-device:
- Queries local REST endpoint `/api/tags` to list installed models (`qwen3:4b`).
- Dispatches chat completion to `/api/chat` and measures local CPU inference throughput (~5.4 tokens/second).
- Confirms that local execution is 100% offline once weights are downloaded, keeping prompts and data strictly on-device.

```powershell
python "Lab 1\activity5.py"
```

### Activity 6 — Use an LLM Through an API / Inference Provider
Calls hosted cloud infrastructure via the Groq SDK (`openai/gpt-oss-20b`):
- Authenticates using `GROQ_API_KEY` stored securely in `.env` (never exposed in code or Git).
- Measures end-to-end cloud latency and token throughput (~60-200+ tokens/second).
- Explains the complete client-server request cycle, authentication, rate limits, and token metering.

```powershell
python "Lab 1\activity6.py"
```

### Activity 7 — Compare Local vs. Colab vs. API
Synthesizes a structured comparison matrix across 8 core dimensions:
- Compares hardware demands, latency, setup difficulty, cost, internet reliance, privacy, scalability, and model size.
- Resolves the **8 GB RAM laptop scenario**: Recommends a Hosted API for daily lab work/fast iteration, Google Colab for GPU/PyTorch layer exploration, and local quantized Ollama (1B-3B) for offline/confidential data.

```powershell
python "Lab 1\activity7.py"
```

---

## 5. Google Colab GPU Instructions (for Activity 3)

To execute `activity3.py` on a cloud GPU:
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime** &rarr; **Change runtime type** &rarr; Select **T4 GPU** &rarr; Click **Save**.
3. Install required libraries in the first cell:
   ```python
   !pip install --quiet torch transformers accelerate
   ```
4. Upload `activity3.py` and run:
   ```python
   !python activity3.py
   ```

---

## 6. Common Errors & Troubleshooting

| Error / Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'transformers'` | Package not yet installed in active Python environment. | Run `pip install transformers` inside your activated virtual environment. |
| `CUDA out of memory` | Model weights and KV cache exceed GPU VRAM. | Use a smaller model, lower batch size, or apply 4-bit quantization (e.g. bitsandbytes / GGUF). |
| `torch.cuda.is_available() returns False` | Running on a machine without an NVIDIA GPU or running CPU PyTorch build. | Normal on integrated Intel graphics. For GPU testing, run `activity3.py` in Google Colab. |
| `RateLimitError` on Hugging Face API | Anonymous API requests to HF Hub exceeded. | Normal; the scripts gracefully fall back to verified offline metadata. |

---

## 7. Viva Preparation & Core Conceptual Summary

1. **Parameter Count vs. Memory Footprint:**  
   Static weight memory is approximately $\text{Parameters} \times \frac{\text{bits}}{8}$. In FP16 (16 bits = 2 bytes), every 1 billion parameters requires $\sim 2 \text{ GB}$ of RAM/VRAM. In INT4 (4 bits = 0.5 byte), 1 billion parameters requires $\sim 0.5 \text{ GB}$.

2. **Why Actual Memory Exceeds Weight Memory:**  
   Real inference requires memory for:
   - Operating System and background services ($\sim 3-4 \text{ GB}$ on Windows).
   - Key-Value (KV) Cache to store attention states for all generated tokens.
   - Activation buffers for intermediate matrix products.
   - Runtime engine context (Ollama / PyTorch thread pools and scratchpads).

3. **Subword Tokenization (BPE):**  
   Language models break text into subwords to maintain a fixed vocabulary size ($\sim 32\text{k} - 128\text{k}$ tokens) while completely avoiding Out-Of-Vocabulary (OOV) errors. Punctuation and spaces are preserved as distinct tokens or prefix tokens.

4. **Why LLMs Are Memory-Bandwidth Bound:**  
   During autoregressive token generation, the model must read all its parameters from memory to generate a single token. Therefore, GPU VRAM ($500 - 1000+\text{ GB/s}$) generates tokens 10x to 50x faster than standard CPU RAM ($40 - 80\text{ GB/s}$).
