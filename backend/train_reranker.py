"""
Train Cross-Encoder Reranker — Binary relevance classification.
Trains on query-document pairs from qa_data.json.

Replaces: Nothing (new component for better retrieval)
Output:   reranker.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

from transformer_models import create_reranker
from train_bpe_tokenizer import BPETokenizer

# ─── Configuration ─────────────────────────────────────────────
QA_DATA_PATH = r"c:\MineLawHub - sandbox\data\training\qa_data.json"
TOKENIZER_PATH = r"c:\MineLawHub - sandbox\backend\bpe_tokenizer.json"
RERANKER_MODEL_PATH = r"c:\MineLawHub - sandbox\backend\reranker.pth"

EPOCHS = 20
BATCH_SIZE = 16
LR = 2e-4
MAX_SEQ_LEN = 128  # Combined query + document length


def clean_text(text):
    """Remove garbled OCR/Hindi text."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_reranker_data(tokenizer, qa_path):
    """
    Create positive and negative pairs for reranker training.
    
    Positive: query + its correct context (label=1)
    Negative: query + random unrelated context (label=0)
    """
    print("  Loading QA data...")
    with open(qa_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Clean and filter
    clean_data = []
    for item in data:
        context = clean_text(item.get("context", ""))
        query = clean_text(item.get("query", ""))
        if len(context) > 20 and len(query) > 5:
            clean_data.append({"query": query, "context": context})
    
    print(f"  Clean pairs: {len(clean_data)}")
    
    # Create training samples
    samples = []
    all_contexts = [d["context"] for d in clean_data]
    
    for item in clean_data:
        query = item["query"]
        pos_context = item["context"]
        
        # Positive pair
        pos_ids = tokenizer.encode_pair(query, pos_context[:200])
        pos_ids = pos_ids[:MAX_SEQ_LEN]
        samples.append((pos_ids, 1.0))
        
        # Negative pair (random context)
        neg_context = random.choice(all_contexts)
        # Make sure it's actually different
        while neg_context == pos_context and len(all_contexts) > 1:
            neg_context = random.choice(all_contexts)
        
        neg_ids = tokenizer.encode_pair(query, neg_context[:200])
        neg_ids = neg_ids[:MAX_SEQ_LEN]
        samples.append((neg_ids, 0.0))
    
    random.shuffle(samples)
    print(f"  Total samples (pos+neg): {len(samples)}")
    return samples


def pad_sequences(sequences, max_len, pad_id=0):
    """Pad sequences and create attention masks."""
    padded = []
    masks = []
    for seq in sequences:
        seq = seq[:max_len]
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_id] * pad_len)
        masks.append([1] * len(seq) + [0] * pad_len)
    return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.float)


def train():
    print("=" * 60)
    print("Phase 3: Training Cross-Encoder Reranker")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/4] Loading BPE tokenizer...")
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    
    # Prepare data
    print("\n[2/4] Preparing training data...")
    samples = prepare_reranker_data(tokenizer, QA_DATA_PATH)
    
    # Create model
    print("\n[3/4] Creating Reranker model...")
    model = create_reranker(tokenizer.vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.BCELoss()
    
    # Training loop
    print(f"\n[4/4] Training for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        random.shuffle(samples)
        total_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, len(samples), BATCH_SIZE):
            batch = samples[i:i + BATCH_SIZE]
            batch_ids = [s[0] for s in batch]
            batch_labels = torch.tensor([s[1] for s in batch], dtype=torch.float)
            
            input_ids, attn_mask = pad_sequences(batch_ids, MAX_SEQ_LEN)
            
            # Forward
            scores = model(input_ids, attn_mask)
            loss = criterion(scores, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = (scores > 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)
        
        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(total // BATCH_SIZE, 1)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1%}")
    
    # Save
    torch.save(model.state_dict(), RERANKER_MODEL_PATH)
    print(f"\n✓ Reranker saved to {RERANKER_MODEL_PATH}")
    print(f"  File size: {os.path.getsize(RERANKER_MODEL_PATH) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    train()
    print("\n✓ Phase 3 complete — Cross-Encoder Reranker trained!")
