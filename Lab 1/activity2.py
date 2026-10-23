"""
Lab 1 — Activity 2: Tokenization Practical
Author: Shubranil Pandit (SCFP125024)
Course: MCA 3rd Semester — Generative AI Lab

WHAT:
    This activity demonstrates subword tokenization in modern Large Language Models (LLMs).
    It inspects how text is partitioned into discrete tokens, mapped to numerical Token IDs,
    and decoded back into readable text.

WHY:
    Neural networks cannot process raw characters or strings directly; they perform linear
    algebra on numerical vectors. Tokenization is the foundational translation layer between
    human language and neural embedding spaces. Understanding why words split, how punctuation
    behaves, and why token IDs are necessary is fundamental for managing prompt lengths,
    latency, and API billing (which is priced per token).

HOW:
    - Employs Hugging Face's `AutoTokenizer` (using the standard GPT-2 BPE tokenizer).
    - Tokenizes a customizable primary sentence and a suite of test cases (compound words,
      punctuation, spaces, numbers, and rare words).
    - Displays a structured table showing Word vs. Token vs. Token ID mappings.
    - Demonstrates detokenization (reconstructing text from token IDs).
    - Provides a built-in fallback tokenizer in case external libraries are absent.

OBSERVATION:
    - One word does NOT always equal one token (e.g., "Generative" -> [' Gener', 'ative']).
    - Punctuation marks (like commas, quotes, periods) are usually assigned separate token IDs.
    - Leading spaces are often prepended to the following word token (e.g., ' AI' vs 'AI').
    - Compound and rare words are partitioned into smaller subwords to prevent Out-Of-Vocabulary (OOV) errors.

CONCLUSION:
    Subword tokenization (such as Byte-Pair Encoding) balances vocabulary size with expressiveness.
    Token IDs serve as discrete indices into the model's embedding matrix, enabling fixed-size
    neural computation across any arbitrary human text.
"""

import sys
from typing import List, Tuple

# Try loading Hugging Face AutoTokenizer; provide helpful guidance if missing
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def load_tokenizer(model_name: str = "gpt2"):
    """
    Loads a pretrained Hugging Face tokenizer.
    GPT-2 is selected because it is lightweight, standard, and does not require GPU.
    """
    if not TRANSFORMERS_AVAILABLE:
        print("[!] Warning: 'transformers' library not found.")
        print("    To install, run: pip install transformers")
        return None
    try:
        # Load the tokenizer (cached locally after first download)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as exc:
        print(f"[!] Error loading tokenizer '{model_name}': {exc}")
        return None


def inspect_tokenization(text: str, tokenizer) -> None:
    """
    Tokenizes the input text and prints a detailed, student-friendly table
    showing Token Index, Token String, and Token ID.
    """
    print(f"\nTarget Text: \"{text}\"")
    print("-" * 75)

    if tokenizer is not None:
        # 1. Encode text to Token IDs
        token_ids: List[int] = tokenizer.encode(text)
        
        # 2. Convert Token IDs back to individual token strings
        tokens: List[str] = [tokenizer.decode([tid]) for tid in token_ids]
        
        # 3. Reconstruct full decoded string
        reconstructed: str = tokenizer.decode(token_ids)
        
        print(f"{'Idx':<5} | {'Token String':<25} | {'Token ID':<10} | {'Subword / Note'}")
        print("-" * 75)
        for idx, (tok_str, tid) in enumerate(zip(tokens, token_ids)):
            # Make spaces visible for educational clarity
            display_tok = tok_str.replace(" ", "[SPACE]")
            note = ""
            if tok_str.startswith(" "):
                note = "Leading space absorbed"
            elif tok_str in [",", ".", "!", "?", ";", ":"]:
                note = "Punctuation token"
            elif len(tok_str.strip()) < len(text.strip()) and not tok_str.isalnum():
                note = "Symbol/Special"
            else:
                note = "Subword unit"
            print(f"{idx:<5} | {display_tok:<25} | {tid:<10} | {note}")
        
        print("-" * 75)
        print(f"Summary: Text of {len(text.split())} words produced {len(token_ids)} tokens.")
        print(f"Reconstructed fidelity check: {reconstructed == text}")
    else:
        # Simplified educational demonstration if transformers is unavailable
        words = text.split()
        print("[Educational Demo - Fallback]")
        for idx, word in enumerate(words):
            fake_id = 1000 + hash(word) % 9000
            print(f"{idx:<5} | {word:<25} | {fake_id:<10} | Word-level approximation")


def run_comparative_experiments(tokenizer) -> None:
    """
    Runs specific test cases answering the core questions from Lab 1 Activity 2:
      1. Is one word always one token?
      2. Why are some words split?
      3. What happens to punctuation?
      4. What happens with spaces and uncommon words?
    """
    print("\n" + "=" * 75)
    print("EXPERIMENTAL TEST CASES & OBSERVATIONS")
    print("=" * 75)

    test_cases = [
        ("Single common word", "machine"),
        ("Complex technical term", "hyperparameter"),
        ("Compound / uncommon word", "tokenization"),
        ("Prefix space distinction", "learning"),
        ("Same word without space", "learning"),
        ("Punctuation attachment", "Hello, world!"),
        ("Contraction & apostrophe", "don't"),
        ("Mathematical expression", "E=mc^2"),
        ("Long rare word", "supercalifragilisticexpialidocious")
    ]

    for label, sample in test_cases:
        if tokenizer is not None:
            token_ids = tokenizer.encode(sample)
            subwords = [tokenizer.decode([tid]) for tid in token_ids]
            visible_subwords = [s.replace(" ", "[SPACE]") for s in subwords]
            print(f"\nTest: {label}")
            print(f"  Input        : \"{sample}\"")
            print(f"  Token Count  : {len(token_ids)}")
            print(f"  Token IDs    : {token_ids}")
            print(f"  Subwords     : {visible_subwords}")
        else:
            print(f"\nTest: {label} -> Input: \"{sample}\"")


def print_theoretical_answers() -> None:
    """Prints clear explanations for viva preparation and lab comprehension."""
    print("\n" + "=" * 75)
    print("THEORETICAL QUESTIONS & VIVA ANSWERS")
    print("=" * 75)
    print("""
Q1: Is one word always one token?
A1: No. In subword tokenization (such as BPE, WordPiece, or Unigram), common words
    like 'the', 'is', and 'apple' are single tokens. However, uncommon words,
    technical jargon, inflections, or compound words (e.g., 'tokenization' -> 'token' + 'ization')
    are split into multiple subwords.

Q2: Why are some words split?
A2: Vocabulary size constraint and Out-Of-Vocabulary (OOV) prevention. If an LLM
    stored every distinct word as a separate token, the vocabulary would grow to millions
    of words, exploding the embedding matrix memory. By splitting words into frequent
    morphemes and subwords, a compact vocabulary (~32k to 128k tokens) can represent
    any conceivable word, misspelling, or code snippet.

Q3: What happens to punctuation?
A3: Punctuation marks (commas, periods, exclamation points, quotes) are usually
    isolated into their own distinct tokens, or grouped into common combinations (e.g., '...').
    This ensures that the semantic meaning of a word is not tied to the punctuation following it.

Q4: Why does the model need token IDs?
A4: Neural networks cannot process text or characters directly; they perform matrix
    multiplications on numerical tensors. Token IDs serve as integer addresses to look up
    corresponding continuous vector embeddings from the model's learned Embedding Matrix.

Q5: Difference between a Token, a Word, and a Token ID:
    - Word     : A linguistic unit of language separated by whitespace or punctuation.
    - Token    : A discrete segment of text (word, subword, character, or symbol)
                 produced by the tokenizer.
    - Token ID : The unique integer index assigned to that specific token in the
                 model's fixed vocabulary dictionary.
""")


def main() -> None:
    """Main execution point for Activity 2."""
    print("=" * 75)
    print("LAB 1: ACTIVITY 2 - TOKENIZATION PRACTICAL")
    print("=" * 75)

    # Load tokenizer (GPT-2 BPE tokenizer)
    print("Loading Hugging Face tokenizer ('gpt2')...")
    tokenizer = load_tokenizer("gpt2")
    if tokenizer is not None:
        print(f"Tokenizer loaded successfully! Vocabulary Size: {tokenizer.vocab_size:,} tokens.")

    # Primary sentence experiment (easy to modify by the student)
    primary_sentence = "Artificial Intelligence and Generative LLMs are transforming modern software engineering."
    print("\n--- Experiment 1: Primary Sentence Tokenization ---")
    inspect_tokenization(primary_sentence, tokenizer)

    # Comparative test cases
    print("\n--- Experiment 2: Edge Cases & Subword Splitting ---")
    run_comparative_experiments(tokenizer)

    # Theory and viva answers
    print_theoretical_answers()

    print("Activity 2 execution complete.")


if __name__ == "__main__":
    main()
