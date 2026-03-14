"""
Custom BPE (Byte Pair Encoding) Tokenizer — Trained from scratch on mining law corpus.
Industry-standard subword tokenization (same algorithm as GPT, BERT, etc.)

Usage:
    python train_bpe_tokenizer.py          # Train and save
    python train_bpe_tokenizer.py --test   # Quick test after training
"""

import json
import re
import os
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

# ─── Configuration ─────────────────────────────────────────────
CORPUS_PATH = r"c:\MineLawHub - sandbox\data\training\corpus.txt"
TOKENIZER_PATH = r"c:\MineLawHub - sandbox\backend\bpe_tokenizer.json"
VOCAB_SIZE = 8000          # Target vocabulary size
MIN_FREQUENCY = 2          # Minimum pair frequency to merge
MAX_CORPUS_CHARS = 2000000 # Limit corpus processing for speed

# Special tokens
SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
    "<SEP>": 4,
    "<CLS>": 5,
    "<MASK>": 6,
}


# ─── Text Preprocessing ───────────────────────────────────────
def clean_corpus_text(text: str) -> str:
    """Clean garbled OCR text and normalize for BPE training."""
    # Remove non-ASCII characters (garbled Hindi OCR)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove (cid:X) artifacts from PDF extraction
    text = re.sub(r'\(cid:\d+\)', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove very short lines (likely garbage)
    lines = text.split('\n')
    lines = [l.strip() for l in lines if len(l.strip()) > 10]
    return ' '.join(lines)


def pre_tokenize(text: str) -> List[str]:
    """
    Split text into words (pre-tokenization step).
    Each word becomes a sequence of characters for BPE to process.
    Preserves word boundaries using GPT-2 style regex pattern.
    """
    # GPT-2 style pattern: splits on word boundaries, keeps punctuation separate
    pattern = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
        re.IGNORECASE
    )
    words = pattern.findall(text.lower())
    return [w for w in words if w.strip()]


def word_to_chars(word: str) -> Tuple[str, ...]:
    """Convert word to tuple of characters (BPE working format)."""
    return tuple(word)


# ─── BPE Training ─────────────────────────────────────────────
def get_pair_counts(word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
    """Count frequency of adjacent symbol pairs across all words."""
    pairs = Counter()
    for word, freq in word_freqs.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq
    return pairs


def merge_pair(word_freqs: Dict[Tuple[str, ...], int], 
               pair: Tuple[str, str]) -> Dict[Tuple[str, ...], int]:
    """Merge all occurrences of a pair in word_freqs."""
    new_word_freqs = {}
    bigram = pair
    replacement = pair[0] + pair[1]
    
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                new_word.append(replacement)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq
    
    return new_word_freqs


def train_bpe(corpus_text: str, vocab_size: int = VOCAB_SIZE) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Train BPE tokenizer from scratch.
    
    Returns:
        vocab: token -> id mapping
        merges: list of merge operations in order
    """
    print("=" * 60)
    print("Training Custom BPE Tokenizer")
    print("=" * 60)
    
    # Step 1: Pre-tokenize corpus into words
    print("\n[1/4] Pre-tokenizing corpus...")
    words = pre_tokenize(corpus_text)
    print(f"  → {len(words):,} word tokens found")
    
    # Step 2: Count word frequencies and convert to character tuples
    word_counts = Counter(words)
    word_freqs = {}
    for word, count in word_counts.items():
        char_tuple = word_to_chars(word)
        if char_tuple:  # Skip empty
            word_freqs[char_tuple] = count
    
    print(f"  → {len(word_freqs):,} unique words")
    
    # Step 3: Build initial character vocabulary
    print("\n[2/4] Building initial character vocabulary...")
    chars = set()
    for word in word_freqs.keys():
        for ch in word:
            chars.add(ch)
    
    # Start vocab with special tokens + all characters
    vocab = dict(SPECIAL_TOKENS)
    next_id = len(SPECIAL_TOKENS)
    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = next_id
            next_id += 1
    
    initial_vocab_size = len(vocab)
    print(f"  → Initial vocab: {initial_vocab_size} tokens (special + characters)")
    
    # Step 4: Iteratively merge most frequent pairs
    num_merges = vocab_size - initial_vocab_size
    print(f"\n[3/4] Learning {num_merges} BPE merges...")
    
    merges = []
    for i in range(num_merges):
        # Count all adjacent pairs
        pair_counts = get_pair_counts(word_freqs)
        
        if not pair_counts:
            print(f"  → No more pairs to merge at step {i}")
            break
        
        # Find most frequent pair
        best_pair = pair_counts.most_common(1)[0]
        pair, freq = best_pair
        
        if freq < MIN_FREQUENCY:
            print(f"  → Stopping at step {i}: best pair frequency ({freq}) below minimum ({MIN_FREQUENCY})")
            break
        
        # Merge the pair
        merged_token = pair[0] + pair[1]
        word_freqs = merge_pair(word_freqs, pair)
        merges.append(pair)
        
        # Add new token to vocab
        if merged_token not in vocab:
            vocab[merged_token] = next_id
            next_id += 1
        
        # Progress logging
        if (i + 1) % 500 == 0 or i < 10:
            print(f"  Merge {i+1:>5}/{num_merges}: '{pair[0]}' + '{pair[1]}' → '{merged_token}' (freq={freq:,})")
    
    print(f"\n[4/4] Final vocabulary: {len(vocab)} tokens, {len(merges)} merges learned")
    
    return vocab, merges


# ─── BPE Tokenizer Class ──────────────────────────────────────
class BPETokenizer:
    """
    Custom BPE Tokenizer for MineLawHub.
    Trained from scratch on mining law corpus — no external dependencies.
    """
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None, 
                 merges: Optional[List[Tuple[str, str]]] = None):
        self.vocab = vocab or {}
        self.merges = merges or []
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Special token IDs
        self.pad_id = SPECIAL_TOKENS["<PAD>"]
        self.sos_id = SPECIAL_TOKENS["<SOS>"]
        self.eos_id = SPECIAL_TOKENS["<EOS>"]
        self.unk_id = SPECIAL_TOKENS["<UNK>"]
        self.sep_id = SPECIAL_TOKENS["<SEP>"]
        self.cls_id = SPECIAL_TOKENS["<CLS>"]
        self.mask_id = SPECIAL_TOKENS["<MASK>"]
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def _apply_merges(self, chars: List[str]) -> List[str]:
        """Apply learned BPE merges to a list of characters."""
        for pair in self.merges:
            i = 0
            new_chars = []
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == pair[0] and chars[i + 1] == pair[1]:
                    new_chars.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        return chars
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to list of token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: If True, prepend <CLS> and append <EOS>
            
        Returns:
            List of integer token IDs
        """
        # Pre-tokenize
        words = pre_tokenize(text.lower())
        
        all_ids = []
        if add_special_tokens:
            all_ids.append(self.cls_id)
        
        for word in words:
            # Split word into characters
            chars = list(word)
            # Apply BPE merges
            tokens = self._apply_merges(chars)
            # Convert to IDs
            for token in tokens:
                token_id = self.vocab.get(token, self.unk_id)
                all_ids.append(token_id)
        
        if add_special_tokens:
            all_ids.append(self.eos_id)
        
        return all_ids
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of integer token IDs
            skip_special: If True, skip special tokens in output
            
        Returns:
            Decoded text string
        """
        special_ids = set(SPECIAL_TOKENS.values()) if skip_special else set()
        tokens = []
        for token_id in ids:
            if token_id in special_ids:
                continue
            token = self.inv_vocab.get(token_id, "<UNK>")
            tokens.append(token)
        return ''.join(tokens)
    
    def encode_pair(self, text_a: str, text_b: str) -> List[int]:
        """
        Encode a pair of texts with [CLS] text_a [SEP] text_b [EOS].
        Used for cross-encoder reranking.
        """
        ids_a = self.encode(text_a)
        ids_b = self.encode(text_b)
        return [self.cls_id] + ids_a + [self.sep_id] + ids_b + [self.eos_id]
    
    def save(self, path: str):
        """Save tokenizer to JSON file."""
        data = {
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "special_tokens": SPECIAL_TOKENS,
            "vocab_size": len(self.vocab),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Tokenizer saved to {path}")
        print(f"  Vocab size: {len(self.vocab)} tokens")
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = data["vocab"]
        merges = [tuple(m) for m in data["merges"]]
        
        tokenizer = cls(vocab=vocab, merges=merges)
        print(f"✓ Tokenizer loaded from {path} ({len(vocab)} tokens)")
        return tokenizer


# ─── Main Training Script ─────────────────────────────────────
def main():
    # Check for test mode
    if "--test" in sys.argv:
        test_tokenizer()
        return
    
    # Load corpus
    print(f"Loading corpus from {CORPUS_PATH}...")
    if not os.path.exists(CORPUS_PATH):
        print(f"ERROR: Corpus not found at {CORPUS_PATH}")
        print("Please ensure corpus.txt exists in data/training/")
        return
    
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    print(f"  Raw corpus: {len(raw_text):,} characters")
    
    # Clean the text
    text = clean_corpus_text(raw_text)
    if len(text) > MAX_CORPUS_CHARS:
        text = text[:MAX_CORPUS_CHARS]
        print(f"  Trimmed to {MAX_CORPUS_CHARS:,} characters for training")
    else:
        print(f"  Cleaned corpus: {len(text):,} characters")
    
    # Train BPE
    vocab, merges = train_bpe(text, vocab_size=VOCAB_SIZE)
    
    # Create tokenizer and save
    tokenizer = BPETokenizer(vocab=vocab, merges=merges)
    tokenizer.save(TOKENIZER_PATH)
    
    # Quick validation
    print("\n" + "=" * 60)
    print("Validation — Sample Encodings")
    print("=" * 60)
    test_texts = [
        "What is Section 40 of the Mines Act?",
        "penalties for contravention of safety regulations",
        "employment of women in underground mines",
        "mineral conservation and development rules 2017",
        "sub-section provisions under coal mines regulation",
    ]
    
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        tokens = [tokenizer.inv_vocab.get(i, '?') for i in ids]
        print(f"\n  Input:   \"{text}\"")
        print(f"  Tokens:  {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        print(f"  IDs:     {ids[:15]}{'...' if len(ids) > 15 else ''}")
        print(f"  Decoded: \"{decoded}\"")
        print(f"  Length:  {len(ids)} tokens")
    
    print("\n✓ BPE Tokenizer training complete!")


def test_tokenizer():
    """Quick test of a pre-trained tokenizer."""
    print("Loading tokenizer for testing...")
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    
    queries = [
        "What is the minimum age to work in mines?",
        "penalties for safety violations",
        "Section 46 employment of women",
        "coal mines regulations 2017 overview",
        "Who is responsible for mine safety?",
    ]
    
    for q in queries:
        ids = tokenizer.encode(q)
        decoded = tokenizer.decode(ids)
        print(f"\n  \"{q}\"")
        print(f"  → {len(ids)} tokens, decoded: \"{decoded}\"")
    
    # Test encode_pair for cross-encoder
    pair_ids = tokenizer.encode_pair("What is Section 40?", "No person below 18 years shall work in mines.")
    print(f"\n  Pair encoding: {len(pair_ids)} tokens")
    print(f"  Decoded: \"{tokenizer.decode(pair_ids)}\"")


if __name__ == "__main__":
    main()
