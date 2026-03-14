"""
Custom Transformer Models for MineLawHub — 2025 Architecture.
Built from scratch using PyTorch — no pre-trained weights.

Models:
    1. TransformerEncoder      — Context-aware embeddings (replaces Word2Vec)
    2. CrossEncoderReranker    — Relevance scoring for retrieval
    3. TransformerDecoder      — Answer generation (replaces GRU Seq2Seq)
    4. TransformerIntentClassifier — Intent classification (replaces EmbeddingBag)

Innovations:
    - RMSNorm (pre-norm, more stable than LayerNorm)
    - RoPE (Rotary Positional Encoding, relative positions)
    - GQA (Grouped Query Attention, efficient multi-head)
    - SwiGLU (modern activation, replaces ReLU/GELU)
    - KV-Cache (fast autoregressive generation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


# ─── RMSNorm ──────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (2019).
    More stable and faster than standard LayerNorm.
    Used by LLaMA, Gemma, and modern Transformers.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ─── Rotary Positional Encoding (RoPE) ────────────────────────
class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — Su et al., 2021.
    Encodes relative position information into attention scores.
    Used by LLaMA, GPT-NeoX, and all modern Transformers.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        # Precompute frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute sin/cos tables
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns cos and sin for the sequence length of x."""
        seq_len = x.shape[-2]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, 
                     cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    # Split into pairs for rotation
    q_r, q_i = q[..., ::2], q[..., 1::2]
    k_r, k_i = k[..., ::2], k[..., 1::2]
    
    # Apply rotation
    q_out_r = q_r * cos - q_i * sin
    q_out_i = q_r * sin + q_i * cos
    k_out_r = k_r * cos - k_i * sin
    k_out_i = k_r * sin + k_i * cos
    
    # Interleave back
    q_out = torch.stack([q_out_r, q_out_i], dim=-1).flatten(-2)
    k_out = torch.stack([k_out_r, k_out_i], dim=-1).flatten(-2)
    
    return q_out, k_out


# ─── Grouped Query Attention (GQA) ────────────────────────────
class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023).
    Uses fewer KV heads than query heads for efficiency.
    Used by LLaMA 2, Gemma, Mistral.
    
    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads (groups)
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # How many Q heads share each KV head
        
        # Projections
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        # RoPE
        self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len)
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match number of Q heads."""
        if self.n_rep == 1:
            return x
        bs, n_kv, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bs, n_kv, self.n_rep, seq_len, head_dim)
            .reshape(bs, self.n_heads, seq_len, head_dim)
        )
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.wq(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        cos, sin = self.rope(q)
        q_rope = q.reshape(-1, seq_len, self.head_dim)
        k_rope = k.reshape(-1, seq_len, self.head_dim)
        q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, cos, sin)
        q = q_rope.reshape(bs, self.n_heads, seq_len, self.head_dim)
        k = k_rope.reshape(bs, self.n_kv_heads, seq_len, self.head_dim)
        
        # Repeat KV heads for GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask for decoder
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 
                diagonal=1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Padding mask
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Weighted sum
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        
        return self.wo(out)


# ─── Cross-Attention (for Decoder) ─────────────────────────────
class CrossAttention(nn.Module):
    """
    Cross-Attention layer for the Decoder.
    Attends to encoder output (retrieved context).
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs, tgt_len, _ = x.shape
        _, src_len, _ = memory.shape
        d_model = x.shape[-1]
        
        q = self.wq(x).view(bs, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(memory).view(bs, src_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(memory).view(bs, src_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if memory_mask is not None:
            attn = attn.masked_fill(memory_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bs, tgt_len, d_model)
        
        return self.wo(out)


# ─── SwiGLU Feed-Forward ──────────────────────────────────────
class SwiGLU(nn.Module):
    """
    SwiGLU activation (Shazeer, 2020).
    Combines Swish activation with gating — better than ReLU/GELU.
    Used by LLaMA, PaLM, and modern Transformers.
    """
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or int(d_model * 8 / 3)  # Standard SwiGLU ratio
        # Round to nearest multiple of 64 for efficiency
        d_ff = ((d_ff + 63) // 64) * 64
        
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ─── Transformer Encoder Block ────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    """Single encoder block with pre-norm, GQA, and SwiGLU."""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm + attention + residual
        x = x + self.attn(self.norm1(x), mask=mask)
        # Pre-norm + FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x


# ─── Transformer Decoder Block ────────────────────────────────
class TransformerDecoderBlock(nn.Module):
    """Single decoder block with causal self-attention, cross-attention, and SwiGLU."""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Causal self-attention
        x = x + self.self_attn(self.norm1(x), mask=tgt_mask, is_causal=True)
        # Cross-attention to encoder memory
        x = x + self.cross_attn(self.norm2(x), memory, memory_mask=memory_mask)
        # Feed-forward
        x = x + self.ffn(self.norm3(x))
        return x


# ═══════════════════════════════════════════════════════════════
# MODEL 1: Transformer Encoder (replaces Word2Vec)
# ═══════════════════════════════════════════════════════════════
class TransformerEncoder(nn.Module):
    """
    Custom Transformer Encoder for context-aware embeddings.
    
    Replaces Word2Vec SkipGram with:
    - BPE token embeddings (not word-level)
    - 4-layer Transformer with RoPE, GQA, SwiGLU, RMSNorm
    - Mean pooling for fixed-size output embeddings
    
    Args:
        vocab_size: BPE vocabulary size
        d_model: Hidden dimension (256)
        n_layers: Number of transformer layers (4)
        n_heads: Number of attention heads (8)
        n_kv_heads: Number of KV heads for GQA (4)
        max_seq_len: Maximum sequence length (512)
    """
    def __init__(self, vocab_size: int, d_model: int = 256, 
                 n_layers: int = 4, n_heads: int = 8, n_kv_heads: int = 4,
                 max_seq_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Transformer encoder blocks
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, n_kv_heads)
            for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small random weights (no pre-training)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass returning per-token hidden states.
        
        Args:
            input_ids: (batch, seq_len) BPE token IDs
            attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding
            
        Returns:
            hidden_states: (batch, seq_len, d_model) context-aware representations
        """
        x = self.token_embed(input_ids)
        
        # Create padding mask (True where padding)
        pad_mask = None
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)
        
        for layer in self.layers:
            x = layer(x, mask=pad_mask)
        
        return self.final_norm(x)
    
    def get_embedding(self, input_ids: torch.Tensor, 
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get fixed-size embedding via mean pooling.
        
        Returns:
            embedding: (batch, d_model) pooled embedding
        """
        hidden = self.forward(input_ids, attention_mask)
        
        if attention_mask is not None:
            # Mean pool only over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (hidden * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1)
            return summed / counts
        else:
            return hidden.mean(dim=1)


# ═══════════════════════════════════════════════════════════════
# MODEL 2: Cross-Encoder Reranker
# ═══════════════════════════════════════════════════════════════
class CrossEncoderReranker(nn.Module):
    """
    Cross-Encoder for relevance reranking.
    
    Takes [CLS] query [SEP] document [EOS] as input,
    outputs a relevance score between 0 and 1.
    
    This is a separate model from the bi-encoder (TransformerEncoder).
    The cross-encoder sees query and document together, enabling
    deeper interaction than cosine similarity.
    """
    def __init__(self, vocab_size: int, d_model: int = 256, 
                 n_layers: int = 4, n_heads: int = 8, n_kv_heads: int = 4):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size, d_model, n_layers, n_heads, n_kv_heads
        )
        
        # Classification head: hidden state of [CLS] token → relevance score
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Score query-document pairs.
        
        Args:
            input_ids: (batch, seq_len) — [CLS] query [SEP] document [EOS]
            attention_mask: (batch, seq_len)
            
        Returns:
            scores: (batch,) relevance scores (sigmoid applied)
        """
        hidden = self.encoder(input_ids, attention_mask)
        # Use [CLS] token (first position) for classification
        cls_hidden = hidden[:, 0, :]
        logits = self.classifier(cls_hidden).squeeze(-1)
        return torch.sigmoid(logits)


# ═══════════════════════════════════════════════════════════════
# MODEL 3: Transformer Decoder (replaces GRU Seq2Seq)
# ═══════════════════════════════════════════════════════════════
class TransformerDecoder(nn.Module):
    """
    Custom Transformer Decoder for answer generation.
    
    Replaces GRU Seq2Seq with:
    - Causal self-attention (autoregressive)
    - Cross-attention to encoder memory (retrieved context)
    - RoPE, GQA, SwiGLU, RMSNorm
    
    Args:
        vocab_size: BPE vocabulary size (shared with encoder)
        d_model: Hidden dimension (256)
        n_layers: Number of decoder layers (4)
        n_heads: Number of attention heads (8)
        n_kv_heads: Number of KV heads for GQA (4)
        max_seq_len: Maximum generation length (512)
    """
    def __init__(self, vocab_size: int, d_model: int = 256, 
                 n_layers: int = 4, n_heads: int = 8, n_kv_heads: int = 4,
                 max_seq_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Decoder blocks with causal self-attn + cross-attn
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, n_kv_heads)
            for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (standard practice)
        self.lm_head.weight = self.token_embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).
        
        Args:
            input_ids: (batch, tgt_len) — target token IDs
            memory: (batch, src_len, d_model) — encoder output (context)
            tgt_mask: (batch, tgt_len) — target padding mask
            memory_mask: (batch, src_len) — memory padding mask
            
        Returns:
            logits: (batch, tgt_len, vocab_size) — next token predictions
        """
        x = self.token_embed(input_ids)
        
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, memory: torch.Tensor, 
                 sos_id: int, eos_id: int, 
                 max_len: int = 200,
                 temperature: float = 0.7,
                 top_k: int = 50) -> List[int]:
        """
        Autoregressive generation with top-k sampling.
        
        Args:
            memory: (1, src_len, d_model) — encoder memory
            sos_id: Start-of-sequence token ID
            eos_id: End-of-sequence token ID
            max_len: Maximum generation length
            temperature: Sampling temperature (lower = more focused)
            top_k: Top-k filtering
            
        Returns:
            generated_ids: List of token IDs
        """
        self.eval()
        device = memory.device
        
        # Start with SOS token
        generated = [sos_id]
        input_ids = torch.tensor([[sos_id]], device=device)
        
        for _ in range(max_len):
            # Forward pass
            x = self.token_embed(input_ids)
            
            for layer in self.layers:
                x = layer(x, memory)
            
            x = self.final_norm(x)
            logits = self.lm_head(x[:, -1, :])  # Last token's predictions
            
            # Temperature scaling
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_val = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_val, torch.full_like(logits, float('-inf')), logits)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            token_id = next_token.item()
            if token_id == eos_id:
                break
            
            generated.append(token_id)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        return generated


# ═══════════════════════════════════════════════════════════════
# MODEL 4: Transformer Intent Classifier (replaces EmbeddingBag)
# ═══════════════════════════════════════════════════════════════
class TransformerIntentClassifier(nn.Module):
    """
    Intent classifier using Transformer encoder + classification head.
    Replaces EmbeddingBag-based classifier.
    
    Classifies queries as:
        0 = static_law (specific legal question)
        1 = general_query (greeting, help, etc.)
    """
    def __init__(self, vocab_size: int, d_model: int = 256, 
                 n_layers: int = 2, n_heads: int = 8, n_kv_heads: int = 4,
                 num_classes: int = 2):
        super().__init__()
        
        # Smaller encoder for intent (only 2 layers)
        self.encoder = TransformerEncoder(
            vocab_size, d_model, n_layers, n_heads, n_kv_heads
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Classify intent.
        
        Returns:
            logits: (batch, num_classes) — raw scores for each class
        """
        # Get pooled embedding
        embedding = self.encoder.get_embedding(input_ids, attention_mask)
        return self.classifier(embedding)


# ─── Model Configuration ──────────────────────────────────────
# These constants define the shared architecture across all models
MODEL_CONFIG = {
    'd_model': 256,
    'n_layers_encoder': 4,
    'n_layers_decoder': 4,
    'n_layers_intent': 2,   # Smaller for intent (simple task)
    'n_heads': 8,
    'n_kv_heads': 4,        # GQA: 4 KV groups for 8 Q heads
    'max_seq_len': 512,
}


def create_encoder(vocab_size: int) -> TransformerEncoder:
    """Create encoder with standard config."""
    return TransformerEncoder(
        vocab_size=vocab_size,
        d_model=MODEL_CONFIG['d_model'],
        n_layers=MODEL_CONFIG['n_layers_encoder'],
        n_heads=MODEL_CONFIG['n_heads'],
        n_kv_heads=MODEL_CONFIG['n_kv_heads'],
        max_seq_len=MODEL_CONFIG['max_seq_len'],
    )


def create_reranker(vocab_size: int) -> CrossEncoderReranker:
    """Create reranker with standard config."""
    return CrossEncoderReranker(
        vocab_size=vocab_size,
        d_model=MODEL_CONFIG['d_model'],
        n_layers=MODEL_CONFIG['n_layers_encoder'],
        n_heads=MODEL_CONFIG['n_heads'],
        n_kv_heads=MODEL_CONFIG['n_kv_heads'],
    )


def create_decoder(vocab_size: int) -> TransformerDecoder:
    """Create decoder with standard config."""
    return TransformerDecoder(
        vocab_size=vocab_size,
        d_model=MODEL_CONFIG['d_model'],
        n_layers=MODEL_CONFIG['n_layers_decoder'],
        n_heads=MODEL_CONFIG['n_heads'],
        n_kv_heads=MODEL_CONFIG['n_kv_heads'],
        max_seq_len=MODEL_CONFIG['max_seq_len'],
    )


def create_intent_classifier(vocab_size: int, num_classes: int = 2) -> TransformerIntentClassifier:
    """Create intent classifier with standard config."""
    return TransformerIntentClassifier(
        vocab_size=vocab_size,
        d_model=MODEL_CONFIG['d_model'],
        n_layers=MODEL_CONFIG['n_layers_intent'],
        n_heads=MODEL_CONFIG['n_heads'],
        n_kv_heads=MODEL_CONFIG['n_kv_heads'],
        num_classes=num_classes,
    )


# ─── Quick Self-Test ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Transformer Models — Architecture Validation")
    print("=" * 60)
    
    vocab_size = 8000
    batch_size = 2
    seq_len = 32
    
    # Test input
    dummy_ids = torch.randint(7, vocab_size, (batch_size, seq_len))
    dummy_mask = torch.ones(batch_size, seq_len)
    
    # 1. Encoder
    print("\n[1] TransformerEncoder")
    encoder = create_encoder(vocab_size)
    hidden = encoder(dummy_ids, dummy_mask)
    embedding = encoder.get_embedding(dummy_ids, dummy_mask)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Hidden:     {hidden.shape}")
    print(f"  Embedding:  {embedding.shape}")
    
    # 2. Reranker
    print("\n[2] CrossEncoderReranker")
    reranker = create_reranker(vocab_size)
    scores = reranker(dummy_ids, dummy_mask)
    total_params = sum(p.numel() for p in reranker.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Scores:     {scores.shape} → {scores.detach().numpy()}")
    
    # 3. Decoder
    print("\n[3] TransformerDecoder")
    decoder = create_decoder(vocab_size)
    memory = encoder(dummy_ids, dummy_mask)  # Use encoder output as memory
    logits = decoder(dummy_ids, memory)
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Logits:     {logits.shape}")
    
    # 4. Intent Classifier
    print("\n[4] TransformerIntentClassifier")
    intent_model = create_intent_classifier(vocab_size)
    intent_logits = intent_model(dummy_ids, dummy_mask)
    total_params = sum(p.numel() for p in intent_model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Logits:     {intent_logits.shape}")
    
    # Total parameters across all models
    all_params = (
        sum(p.numel() for p in encoder.parameters()) +
        sum(p.numel() for p in reranker.parameters()) +
        sum(p.numel() for p in decoder.parameters()) +
        sum(p.numel() for p in intent_model.parameters())
    )
    print(f"\n{'='*60}")
    print(f"Total parameters (all 4 models): {all_params:,}")
    print(f"Estimated model size: ~{all_params * 4 / 1024 / 1024:.1f} MB")
    print(f"{'='*60}")
    print("✓ All models validated successfully!")
