"""
Train Transformer Encoder — Contrastive Learning on mining law corpus.
Also trains the intent classifier using intent_data.json.

Replaces: train_embeddings.py (Word2Vec SkipGram)
Output:   transformer_encoder.pth, transformer_intent.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import random
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from transformer_models import create_encoder, create_intent_classifier
from train_bpe_tokenizer import BPETokenizer

# ─── Configuration ─────────────────────────────────────────────
CORPUS_PATH = r"c:\MineLawHub - sandbox\data\training\corpus.txt"
INTENT_DATA_PATH = r"c:\MineLawHub - sandbox\data\training\intent_data.json"
TOKENIZER_PATH = r"c:\MineLawHub - sandbox\backend\bpe_tokenizer.json"
ENCODER_MODEL_PATH = r"c:\MineLawHub - sandbox\backend\transformer_encoder.pth"
INTENT_MODEL_PATH = r"c:\MineLawHub - sandbox\backend\transformer_intent.pth"

# Encoder training hyperparams
ENCODER_EPOCHS = 30
ENCODER_BATCH_SIZE = 32
ENCODER_LR = 3e-4
CHUNK_SIZE = 64          # Tokens per chunk for contrastive pairs
OVERLAP_SIZE = 16        # Token overlap for positive pairs
MAX_CHUNKS = 5000        # Max training chunks
TEMPERATURE = 0.07       # InfoNCE temperature

# Intent training hyperparams
INTENT_EPOCHS = 30
INTENT_BATCH_SIZE = 16
INTENT_LR = 1e-4
MAX_SEQ_LEN = 64


# ─── Data Preparation ─────────────────────────────────────────
def prepare_contrastive_chunks(tokenizer, corpus_path, chunk_size, overlap, max_chunks):
    """
    Create contrastive learning pairs from corpus.
    
    Positive pairs: Overlapping windows from the same text region
    Negative pairs: Random chunks from different regions
    """
    import re
    
    print("  Loading and cleaning corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Clean garbled text (same as BPE training)
    text = re.sub(r'[^\x00-\x7F]+', ' ', raw_text)
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    print(f"  Cleaned corpus: {len(text):,} characters")
    
    # Tokenize entire corpus
    print("  Tokenizing corpus (this may take a moment)...")
    all_ids = tokenizer.encode(text[:500000])  # Limit for speed
    print(f"  Total tokens: {len(all_ids):,}")
    
    # Create overlapping chunks
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(all_ids) - chunk_size, step):
        chunk = all_ids[i:i + chunk_size]
        chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break
    
    print(f"  Created {len(chunks):,} chunks (size={chunk_size}, overlap={overlap})")
    
    # Create positive pairs (adjacent/overlapping chunks)
    positive_pairs = []
    for i in range(len(chunks) - 1):
        positive_pairs.append((chunks[i], chunks[i + 1]))
    
    # Shuffle and limit
    random.shuffle(positive_pairs)
    positive_pairs = positive_pairs[:max_chunks]
    
    print(f"  Positive pairs: {len(positive_pairs):,}")
    return chunks, positive_pairs


def pad_sequences(sequences, max_len, pad_id=0):
    """Pad sequences to max_len and create attention masks."""
    padded = []
    masks = []
    for seq in sequences:
        seq = seq[:max_len]
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_id] * pad_len)
        masks.append([1] * len(seq) + [0] * pad_len)
    return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.float)


# ─── Contrastive Loss (InfoNCE) ───────────────────────────────
def infonce_loss(embeddings_a, embeddings_b, temperature=TEMPERATURE):
    """
    InfoNCE contrastive loss.
    Each (a_i, b_i) pair is positive; all other combinations are negative.
    """
    # Normalize embeddings
    a = F.normalize(embeddings_a, dim=-1)
    b = F.normalize(embeddings_b, dim=-1)
    
    # Similarity matrix
    logits = torch.matmul(a, b.T) / temperature
    
    # Labels: diagonal = positive pairs
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # Symmetric loss
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    
    return (loss_a + loss_b) / 2


# ─── Train Encoder ─────────────────────────────────────────────
def train_encoder():
    """Train Transformer Encoder with contrastive learning."""
    print("=" * 60)
    print("Phase 2A: Training Transformer Encoder")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/4] Loading BPE tokenizer...")
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    
    # Prepare data
    print("\n[2/4] Preparing contrastive training data...")
    chunks, positive_pairs = prepare_contrastive_chunks(
        tokenizer, CORPUS_PATH, CHUNK_SIZE, OVERLAP_SIZE, MAX_CHUNKS
    )
    
    # Create model
    print("\n[3/4] Creating Transformer Encoder...")
    encoder = create_encoder(tokenizer.vocab_size)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {total_params:,}")
    
    optimizer = optim.AdamW(encoder.parameters(), lr=ENCODER_LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ENCODER_EPOCHS)
    
    # Training loop
    print(f"\n[4/4] Training for {ENCODER_EPOCHS} epochs...")
    encoder.train()
    
    for epoch in range(ENCODER_EPOCHS):
        random.shuffle(positive_pairs)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(positive_pairs), ENCODER_BATCH_SIZE):
            batch_pairs = positive_pairs[i:i + ENCODER_BATCH_SIZE]
            
            anchors = [p[0] for p in batch_pairs]
            positives = [p[1] for p in batch_pairs]
            
            # Pad and create tensors
            anchor_ids, anchor_mask = pad_sequences(anchors, CHUNK_SIZE)
            pos_ids, pos_mask = pad_sequences(positives, CHUNK_SIZE)
            
            # Get embeddings
            anchor_emb = encoder.get_embedding(anchor_ids, anchor_mask)
            pos_emb = encoder.get_embedding(pos_ids, pos_mask)
            
            # Compute contrastive loss
            loss = infonce_loss(anchor_emb, pos_emb)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{ENCODER_EPOCHS} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save
    torch.save(encoder.state_dict(), ENCODER_MODEL_PATH)
    print(f"\n✓ Transformer Encoder saved to {ENCODER_MODEL_PATH}")
    print(f"  File size: {os.path.getsize(ENCODER_MODEL_PATH) / 1024 / 1024:.1f} MB")
    
    return encoder


# ─── Train Intent Classifier ──────────────────────────────────
def train_intent_classifier():
    """Train Transformer Intent Classifier."""
    print("\n" + "=" * 60)
    print("Phase 2B: Training Intent Classifier")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/4] Loading BPE tokenizer...")
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    
    # Load intent data
    print("\n[2/4] Loading intent training data...")
    with open(INTENT_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    label_map = {"static_law": 0, "general_query": 1}
    
    # Tokenize all samples
    all_ids = []
    all_labels = []
    for item in data:
        ids = tokenizer.encode(item["text"])[:MAX_SEQ_LEN]
        label = label_map[item["label"]]
        all_ids.append(ids)
        all_labels.append(label)
    
    print(f"  Samples: {len(all_ids)}")
    print(f"  Labels: {dict((v, sum(1 for l in all_labels if l == label_map[v])) for v in label_map)}")
    
    # Create model
    print("\n[3/4] Creating Intent Classifier...")
    model = create_intent_classifier(tokenizer.vocab_size, num_classes=len(label_map))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=INTENT_LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\n[4/4] Training for {INTENT_EPOCHS} epochs...")
    model.train()
    
    indices = list(range(len(all_ids)))
    
    for epoch in range(INTENT_EPOCHS):
        random.shuffle(indices)
        total_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, len(indices), INTENT_BATCH_SIZE):
            batch_idx = indices[i:i + INTENT_BATCH_SIZE]
            batch_ids = [all_ids[j] for j in batch_idx]
            batch_labels = torch.tensor([all_labels[j] for j in batch_idx], dtype=torch.long)
            
            # Pad
            input_ids, attn_mask = pad_sequences(batch_ids, MAX_SEQ_LEN)
            
            # Forward
            logits = model(input_ids, attn_mask)
            loss = criterion(logits, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)
        
        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(total // INTENT_BATCH_SIZE, 1)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{INTENT_EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1%}")
    
    # Save
    torch.save(model.state_dict(), INTENT_MODEL_PATH)
    print(f"\n✓ Intent Classifier saved to {INTENT_MODEL_PATH}")
    print(f"  File size: {os.path.getsize(INTENT_MODEL_PATH) / 1024 / 1024:.1f} MB")


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    train_encoder()
    train_intent_classifier()
    print("\n" + "=" * 60)
    print("✓ Phase 2 complete — Encoder + Intent Classifier trained!")
    print("=" * 60)
