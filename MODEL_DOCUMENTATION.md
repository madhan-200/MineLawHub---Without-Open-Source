# MineLawHub — Custom Transformer Models Documentation

> **For Team Review:** This document explains every custom AI model in our project.
> Read this before the review so you can confidently explain how everything works.

---

## Quick Summary (Read This First)

We built **4 custom Transformer neural networks** from scratch using PyTorch:

| # | Model | File | Parameters | What It Does |
|---|-------|------|-----------|--------------|
| 1 | **Transformer Encoder** | `transformer_encoder.pth` | 5M | Converts text → 256-dim number vectors |
| 2 | **Cross-Encoder Reranker** | `reranker.pth` | 5M | Scores how relevant a document is to a query |
| 3 | **Transformer Decoder** | `transformer_decoder.pth` | 6M | Generates human-readable answers |
| 4 | **Intent Classifier** | `transformer_intent.pth` | 3.5M | Detects if query is legal question or greeting |

**Total: 19.6M parameters. All trained from scratch. Zero pre-trained weights. Zero external APIs.**

---

## How a Query is Processed (Step by Step)

```
User types: "What is the minimum age to work in mines?"
                    │
                    ▼
        ┌──────────────────────┐
   [1]  │   BPE TOKENIZER      │  Splits text into subwords
        │                      │  "mining" → ["min", "ing"]
        └──────────┬───────────┘  Output: [token IDs]
                    │
                    ▼
        ┌──────────────────────┐
   [2]  │  INTENT CLASSIFIER   │  Is this a legal question or greeting?
        │  (Model 4)           │  Output: "static_law" or "general_query"
        └──────────┬───────────┘
                    │
                    ▼
        ┌──────────────────────┐
   [3]  │  TRANSFORMER ENCODER │  Converts query → 256-dim embedding vector
        │  (Model 1)           │  This vector captures the MEANING of the query
        └──────────┬───────────┘
                    │
                    ▼
        ┌──────────────────────┐
   [4]  │  CHROMADB SEARCH     │  Finds most similar legal sections
        │  (Hybrid Search)     │  Uses embedding vector + keyword matching
        └──────────┬───────────┘
                    │
                    ▼
        ┌──────────────────────┐
   [5]  │  CROSS-ENCODER       │  Re-scores each result by reading
        │  RERANKER (Model 2)  │  query + document TOGETHER
        └──────────┬───────────┘  Output: Most relevant sections
                    │
                    ▼
        ┌──────────────────────┐
   [6]  │  TRANSFORMER DECODER │  Reads the query + retrieved sections
        │  (Model 3)           │  Generates a natural language answer
        └──────────┬───────────┘
                    │
                    ▼
            Final Answer shown to user
```

---

## The BPE Tokenizer (Not a Model, But Important)

### What is it?
BPE (Byte Pair Encoding) is how we split text into tokens. 

### Why not just split by spaces?
Splitting by spaces fails for unknown words:
```
Space splitting: "contraventions" → ["contraventions"]  ← UNKNOWN WORD!
BPE splitting:   "contraventions" → ["contra", "vent", "ations"]  ← WORKS!
```

### How does it work?
1. Start with individual characters: `["m", "i", "n", "i", "n", "g"]`
2. Find the most frequent pair: `("i", "n")` appears most → merge into `"in"`
3. Now: `["m", "in", "in", "g"]`  
4. Find next frequent pair: `("in", "g")` → merge into `"ing"`
5. Now: `["m", "in", "ing"]`
6. Repeat until we have 8,000 subwords

### Key Facts for Review
- **Vocabulary size**: 8,000 subword tokens
- **Special tokens**: `<PAD>` (0), `<UNK>` (1), `<SOS>` (2), `<EOS>` (3), `<CLS>` (4), `<SEP>` (5)
- **Trained on**: Our mining law corpus (corpus.txt)
- **File**: `backend/train_bpe_tokenizer.py`
- **Output**: `backend/bpe_tokenizer.json`
- **Industry standard**: Same method used by GPT, BERT, LLaMA, etc.

---

## Model 1: Transformer Encoder

### What does it do?
Converts text into a **256-dimensional number vector** (called an "embedding"). This vector captures the **meaning** of the text, not just the words.

### Simple Analogy
Think of it like translating text into a secret code where similar meanings have similar codes:
- "age limit in mines" → `[0.82, -0.15, 0.44, ...]` (256 numbers)
- "minimum working age" → `[0.79, -0.12, 0.41, ...]` (similar numbers!)
- "coal price" → `[-0.31, 0.67, -0.55, ...]` (very different numbers)

### Architecture (Explain This in Review)
```
Input: Token IDs [45, 823, 156, 92, ...]
                    │
                    ▼
        ┌──────────────────────┐
        │  Token Embedding     │  Each token → 256-dim vector
        │  (8000 × 256 table)  │
        └──────────┬───────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  Transformer Layer 1 │  ← Each layer has:
        │  Transformer Layer 2 │     • RMSNorm (normalization)
        │  Transformer Layer 3 │     • GQA (attention mechanism)
        │  Transformer Layer 4 │     • SwiGLU (feed-forward)
        └──────────┬───────────┘     • Residual connections
                    │
                    ▼
        ┌──────────────────────┐
        │  Mean Pooling        │  Average all token vectors → 1 vector
        └──────────┬───────────┘
                    │
                    ▼
        Output: [256-dim embedding]
```

### Key Facts for Review
- **Parameters**: 4,999,424 (~5M)
- **Layers**: 4 Transformer layers
- **Attention heads**: 8 query heads, 4 KV heads (GQA)
- **Embedding dimension**: 256
- **Training method**: Contrastive learning (InfoNCE loss)
- **Training data**: corpus.txt
- **Training result**: Loss dropped from 1.64 → 0.04 (30 epochs)
- **File**: `backend/transformer_models.py` → class `TransformerEncoder`
- **Training script**: `backend/train_transformer_encoder.py`

### What is Contrastive Learning? (Review Question)
We train the encoder to:
- Make **similar texts** have **similar embeddings** (close together)
- Make **different texts** have **different embeddings** (far apart)

Example:
```
"What is Section 40?"  ←→  "Section 40 of Mines Act"     → CLOSE (similar meaning)
"What is Section 40?"  ←→  "Coal prices in 2024"         → FAR (different meaning)
```

---

## Model 2: Cross-Encoder Reranker

### What does it do?
After ChromaDB retrieves 5 candidate documents, the **Reranker scores each one** to find the MOST relevant document. It reads the query and document TOGETHER for deeper understanding.

### Why is it needed?
The Encoder (Model 1) compares query and document SEPARATELY:
```
Encoder:   query → vector1    document → vector2    compare vectors (fast but shallow)
Reranker:  [query + document together] → single relevance score (slower but much more accurate)
```

### Architecture
```
Input: "[CLS] What is Section 40? [SEP] Section 40 says no person below 18... [EOS]"
              ↑ query part              ↑ document part
                    │
                    ▼
        ┌──────────────────────┐
        │  Transformer Encoder │  Same architecture as Model 1
        │  (4 layers)          │  But sees query+doc TOGETHER
        └──────────┬───────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  Take [CLS] token    │  The first token's vector
        │  Classification Head │  Linear → SiLU → Dropout → Linear
        │  Sigmoid             │
        └──────────┬───────────┘
                    │
                    ▼
        Output: 0.95 (highly relevant) or 0.12 (not relevant)
```

### Key Facts for Review
- **Parameters**: 5,032,449 (~5M)
- **Training method**: Binary classification (relevant=1, irrelevant=0)
- **Training data**: qa_data.json (positive pairs) + random negative pairs
- **Training result**: 98.7% accuracy
- **File**: `backend/transformer_models.py` → class `CrossEncoderReranker`
- **Training script**: `backend/train_reranker.py`

---

## Model 3: Transformer Decoder

### What does it do?
**Generates the final answer** word by word (token by token). It reads the retrieved legal sections and produces a human-readable answer.

### Simple Analogy
Like autocomplete on your phone, but much smarter — it predicts the next word based on the question AND the legal context.

### Architecture
```
Input: [SOS] token1 token2 ...    (answer being generated)
  +
Memory: [encoder output of retrieved legal sections]  (context to read from)
                    │
                    ▼
        ┌──────────────────────────────────┐
        │  Decoder Layer 1-4, each with:   │
        │                                  │
        │  1. Causal Self-Attention        │  ← Looks at answer so far
        │     (can only look LEFT,         │     (can't look at future words)
        │      not at future tokens)       │
        │                                  │
        │  2. Cross-Attention              │  ← Looks at the legal sections
        │     (reads the context/memory)   │     (finds relevant info)
        │                                  │
        │  3. SwiGLU Feed-Forward          │  ← Processes the information
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────┐
        │  Linear → Softmax    │  → Predicts next token
        │  (Weight Tying)      │     from 8000 vocabulary
        └──────────┬───────────┘
                    │
                    ▼
        Output: Next token. Repeat until [EOS] or 200 tokens.
```

### What is Weight Tying? (Review Question)
The input embedding table and output prediction layer **share the same weights**. This:
- Reduces parameters (saves memory)
- Improves training (the model learns better word representations)
- Industry standard since 2016

### Key Facts for Review
- **Parameters**: 6,049,024 (~6M)
- **Layers**: 4 decoder layers
- **Max generation**: 200 tokens
- **Training method**: Next-token prediction (teacher forcing)
- **Training data**: qa_data.json (question → answer pairs)
- **Training result**: Loss dropped from 8.1 → 1.96 (30 epochs)
- **Generation**: Top-k sampling with temperature=0.7
- **File**: `backend/transformer_models.py` → class `TransformerDecoder`
- **Training script**: `backend/train_decoder.py`

---

## Model 4: Intent Classifier

### What does it do?
Classifies user input as:
- **`static_law`** → Legal question (search the database)
- **`general_query`** → Greeting/chitchat (respond with a greeting)

### Why is it needed?
If someone says "Hello!", we shouldn't search the law database. If someone asks "What is Section 40?", we should.

### Architecture
```
Input: "Hello how are you?"  or  "What is Section 40?"
                    │
                    ▼
        ┌──────────────────────┐
        │  Transformer Encoder │  Same architecture as Model 1
        │  (but only 2 layers) │  Smaller because task is simple
        └──────────┬───────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  Mean Pooling        │  → 256-dim vector
        │  Classification Head │  Linear → SiLU → Dropout → Linear
        └──────────┬───────────┘
                    │
                    ▼
        Output: [0.98, 0.02] → "static_law"  (legal question)
            or: [0.05, 0.95] → "general_query" (greeting)
```

### Key Facts for Review
- **Parameters**: 3,556,994 (~3.5M)
- **Layers**: 2 Transformer layers (simpler task needs fewer layers)
- **Classes**: 2 (static_law, general_query)
- **Training method**: Supervised classification (cross-entropy loss)
- **Training data**: intent_data.json
- **Training result**: **100% accuracy**
- **File**: `backend/transformer_models.py` → class `TransformerIntentClassifier`
- **Training script**: `backend/train_transformer_encoder.py` (trained together with encoder)

---

## The 5 Modern Innovations (Know These for Review)

### 1. RMSNorm (Root Mean Square Normalization)
**What**: Normalizes layer outputs to prevent values from growing too large or small.  
**Why better than LayerNorm**: Faster (no mean calculation) and more stable.  
**Used by**: LLaMA, Gemma, modern Transformers.  
**Our code**: `class RMSNorm` in `transformer_models.py`  

### 2. RoPE (Rotary Positional Encoding)
**What**: Encodes the position of each token using rotation in vector space.  
**Why better than absolute positions**: Captures **relative** distance between tokens. "Section 40" understands that "Section" is next to "40" regardless of where they appear in the sentence.  
**Used by**: LLaMA, GPT-NeoX, all modern models.  
**Our code**: `class RotaryPositionalEncoding` in `transformer_models.py`  

### 3. GQA (Grouped Query Attention)
**What**: Uses **8 query heads** but only **4 key-value heads**. Each KV head is shared by 2 query heads.  
**Why better than standard multi-head**: Uses less memory and is faster, with almost no accuracy loss.  
**Used by**: LLaMA 2, Gemma, Mistral.  
**Our code**: `class GroupedQueryAttention` in `transformer_models.py`  

### 4. SwiGLU (Swish-Gated Linear Unit)
**What**: Replaces the standard ReLU/GELU activation in the feed-forward network.  
**How it works**: `SwiGLU(x) = Swish(W1 · x) ⊙ (W3 · x)` then `W2 · result`  
**Why better**: Shows consistently better training than ReLU/GELU in research.  
**Used by**: LLaMA, PaLM, Gemini.  
**Our code**: `class SwiGLU` in `transformer_models.py`  

### 5. BPE Tokenizer (Byte Pair Encoding)
**What**: Breaks text into subword pieces instead of whole words.  
**Why better than word-level**: No unknown words ever. Small vocabulary (8,000 vs 50,000+).  
**Used by**: GPT, BERT, LLaMA, every modern model.  
**Our code**: `class BPETokenizer` in `train_bpe_tokenizer.py`  

---

## Common Review Questions & Answers

### Q: Why not use ChatGPT/Gemini API?
**A:** Our ma'am's requirement — no external APIs, no open-source models. Everything custom-built. Also, using APIs means data leaves your machine, which is a privacy concern for legal queries.

### Q: Why 256 dimensions and not 768 like BERT?
**A:** Our dataset is domain-specific (mining laws only), not general knowledge. 256 dimensions is enough to capture the patterns in our data while keeping the model small and fast.

### Q: How is this different from the old Word2Vec approach?
**A:** Word2Vec creates one fixed vector per word. "mine" always has the same vector whether it means "a coal mine" or "belongs to me". Our Transformer Encoder creates **context-aware** vectors — the same word gets different vectors based on surrounding words.

### Q: What is "training from scratch"?
**A:** We start with random weights (random numbers) and train the model by showing it our mining law data. The model learns patterns from scratch. We do NOT download any pre-trained weights from the internet.

### Q: How does the Decoder generate answers if it was trained from scratch?
**A:** We trained it on our qa_data.json which has question-answer pairs about mining laws. The model learned to predict the next word in an answer given a question and context. It's not as fluent as GPT-4, but it generates relevant legal answers for our domain.

### Q: What is the total model size on disk?
**A:** ~75 MB total (all 4 model files + tokenizer). Very lightweight — runs on any laptop without GPU.

---

## File Reference

| File | What It Contains |
|------|-----------------|
| `backend/transformer_models.py` | All 4 model architectures (PyTorch classes) |
| `backend/train_bpe_tokenizer.py` | BPE tokenizer training + BPETokenizer class |
| `backend/train_transformer_encoder.py` | Trains Encoder (contrastive) + Intent (supervised) |
| `backend/train_reranker.py` | Trains Cross-Encoder Reranker |
| `backend/train_decoder.py` | Trains Transformer Decoder |
| `backend/custom_client.py` | Loads all models and runs the inference pipeline |
| `backend/search_engine.py` | ChromaDB hybrid search |
| `backend/main.py` | FastAPI endpoints |
