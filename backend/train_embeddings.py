import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import pickle
import os
import random

# Configuration
CORPUS_PATH = r"c:\MineLawHub - sandbox\data\training\corpus.txt"
MODEL_PATH = r"c:\MineLawHub - sandbox\backend\word2vec.pth"
VOCAB_PATH = r"c:\MineLawHub - sandbox\backend\vocab_w2v.pkl"
EMBED_DIM = 128
WINDOW_SIZE = 5
EPOCHS = 100
BATCH_SIZE = 128
MAX_PAIRS = 50000 
NGRAM_BUCKETS = 5000 # Hash bucket size for subwords

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, n_gram_buckets, embed_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.ngram_embeddings = nn.Embedding(n_gram_buckets, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, target, ngrams):
        # Sum word embedding and mean of its n-gram embeddings
        v_word = self.embeddings(target)
        v_ngram = torch.mean(self.ngram_embeddings(ngrams), dim=1)
        v = v_word + v_ngram
        return self.output(v)

def get_ngrams(word, n=3):
    """Generate character n-grams for a word."""
    word = f"<{word}>"
    if len(word) < n:
        return [hash(word) % NGRAM_BUCKETS]
    return [hash(word[i:i+n]) % NGRAM_BUCKETS for i in range(len(word)-n+1)]

def train():
    if not os.path.exists(CORPUS_PATH):
        print("Corpus not found. Run prepare_dataset.py first.")
        return

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read().lower().split()

    # Build Vocab
    word_counts = Counter(text)
    filtered_words = [word for word, count in word_counts.items() if count > 2]
    vocab = {word: i for i, word in enumerate(filtered_words)}
    vocab["<UNK>"] = len(vocab)
    
    # Pre-calculate n-grams for each word in vocab
    vocab_ngrams = {}
    for word, idx in vocab.items():
        ngrams = get_ngrams(word)
        # Pad ngrams to same length for batching
        vocab_ngrams[idx] = ngrams[:8] + [0] * (8 - len(ngrams[:8]))
    
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)

    # Generate Training Pairs
    pairs = []
    print("Generating training pairs...")
    for i, word in enumerate(text):
        if word not in vocab: continue
        target_idx = vocab[word]
        for j in range(max(0, i - WINDOW_SIZE), min(len(text), i + WINDOW_SIZE + 1)):
            if i == j: continue
            context_word = text[j]
            if context_word in vocab:
                pairs.append((target_idx, vocab[context_word]))
    
    print(f"Total pairs: {len(pairs)}")
    if len(pairs) > MAX_PAIRS:
        print(f"Limiting to {MAX_PAIRS} pairs...")
        pairs = random.sample(pairs, MAX_PAIRS)
    
    # Model
    model = SkipGramModel(len(vocab), NGRAM_BUCKETS, EMBED_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Simple DataLoader alternative
    def get_batches(pairs, batch_size):
        random.shuffle(pairs)
        for i in range(0, len(pairs), batch_size):
            yield pairs[i:i+batch_size]

    print("Training Subword-Aware Word2Vec...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in get_batches(pairs, BATCH_SIZE):
            targets = torch.tensor([p[0] for p in batch])
            contexts = torch.tensor([p[1] for p in batch])
            # Get n-grams for targets
            batch_ngrams = torch.tensor([vocab_ngrams[p[0]] for p in batch])
            
            optimizer.zero_grad()
            outputs = model(targets, batch_ngrams)
            loss = criterion(outputs, contexts)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(pairs):.4f}")
            
    torch.save(model.state_dict(), MODEL_PATH)
    print("Subword embeddings trained and saved!")

if __name__ == "__main__":
    train()
