"""
Train Transformer Decoder — Next-token prediction with cross-attention.
Trained on QA pairs from qa_data.json.

Replaces: train_generator.py (GRU Seq2Seq)
Output:   transformer_decoder.pth
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

from transformer_models import create_encoder, create_decoder
from train_bpe_tokenizer import BPETokenizer

# ─── Configuration ─────────────────────────────────────────────
QA_DATA_PATH = r"c:\MineLawHub - sandbox\data\training\qa_data.json"
TOKENIZER_PATH = r"c:\MineLawHub - sandbox\backend\bpe_tokenizer.json"
ENCODER_MODEL_PATH = r"c:\MineLawHub - sandbox\backend\transformer_encoder.pth"
DECODER_MODEL_PATH = r"c:\MineLawHub - sandbox\backend\transformer_decoder.pth"

EPOCHS = 30
BATCH_SIZE = 8
LR = 3e-4
MAX_CONTEXT_LEN = 128   # Encoder input (context) max tokens
MAX_ANSWER_LEN = 128    # Decoder output (answer) max tokens


def clean_text(text):
    """Remove garbled OCR/Hindi text."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_decoder_data(tokenizer, qa_path):
    """
    Prepare training data for decoder.
    
    Each sample: (context_ids, answer_ids)
    - context_ids: BPE-encoded context chunk (encoder input)
    - answer_ids: BPE-encoded answer with SOS/EOS (decoder target)
    """
    print("  Loading QA data...")
    with open(qa_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    skipped = 0
    
    for item in data:
        # Clean both context and answer
        context = clean_text(item.get("context", ""))
        answer = clean_text(item.get("answer", ""))
        query = clean_text(item.get("query", ""))
        
        # Skip entries with too little content
        if len(context) < 20 or len(answer) < 10:
            skipped += 1
            continue
        
        # Combine query + context for encoder input
        combined_context = f"{query} {context[:300]}"
        
        # Tokenize
        context_ids = tokenizer.encode(combined_context)[:MAX_CONTEXT_LEN]
        
        # Answer with SOS and EOS
        answer_ids = [tokenizer.sos_id] + tokenizer.encode(answer[:300])[:MAX_ANSWER_LEN - 2] + [tokenizer.eos_id]
        
        samples.append((context_ids, answer_ids))
    
    print(f"  Valid samples: {len(samples)}, skipped: {skipped}")
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
    print("Phase 4: Training Transformer Decoder")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/5] Loading BPE tokenizer...")
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    
    # Prepare data
    print("\n[2/5] Preparing training data...")
    samples = prepare_decoder_data(tokenizer, QA_DATA_PATH)
    
    # Load pre-trained encoder (frozen — used to produce memory for cross-attention)
    print("\n[3/5] Loading pre-trained Transformer Encoder...")
    encoder = create_encoder(tokenizer.vocab_size)
    if os.path.exists(ENCODER_MODEL_PATH):
        encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, weights_only=True))
        print(f"  ✓ Loaded encoder from {ENCODER_MODEL_PATH}")
    else:
        print(f"  ⚠ Encoder not found at {ENCODER_MODEL_PATH}, using random init")
    encoder.eval()  # Freeze encoder during decoder training
    
    # Create decoder
    print("\n[4/5] Creating Transformer Decoder...")
    decoder = create_decoder(tokenizer.vocab_size)
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Parameters: {total_params:,}")
    
    optimizer = optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Training loop
    print(f"\n[5/5] Training for {EPOCHS} epochs...")
    decoder.train()
    
    for epoch in range(EPOCHS):
        random.shuffle(samples)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(samples), BATCH_SIZE):
            batch = samples[i:i + BATCH_SIZE]
            
            context_seqs = [s[0] for s in batch]
            answer_seqs = [s[1] for s in batch]
            
            # Pad context (encoder input)
            ctx_ids, ctx_mask = pad_sequences(context_seqs, MAX_CONTEXT_LEN)
            
            # Pad answers (decoder input/target)
            ans_ids, ans_mask = pad_sequences(answer_seqs, MAX_ANSWER_LEN)
            
            # Get encoder memory (no gradient — encoder is frozen)
            with torch.no_grad():
                memory = encoder(ctx_ids, ctx_mask)
            
            # Decoder input: answer tokens shifted right (teacher forcing)
            # Input: [SOS, tok1, tok2, ...] → Target: [tok1, tok2, ..., EOS]
            dec_input = ans_ids[:, :-1]   # All but last
            dec_target = ans_ids[:, 1:]   # All but first
            
            # Create memory mask (True where padding in context)
            memory_mask = (ctx_mask == 0)
            
            # Forward
            logits = decoder(dec_input, memory, memory_mask=memory_mask)
            
            # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                dec_target.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save
    torch.save(decoder.state_dict(), DECODER_MODEL_PATH)
    print(f"\n✓ Transformer Decoder saved to {DECODER_MODEL_PATH}")
    print(f"  File size: {os.path.getsize(DECODER_MODEL_PATH) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    train()
    print("\n✓ Phase 4 complete — Transformer Decoder trained!")
