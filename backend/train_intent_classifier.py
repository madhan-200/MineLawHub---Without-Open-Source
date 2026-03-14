import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from collections import Counter
import pickle

# Configuration
DATA_PATH = r"c:\MineLawHub - sandbox\data\training\intent_data.json"
MODEL_PATH = r"c:\MineLawHub - sandbox\backend\intent_classifier.pth"
VOCAB_PATH = r"c:\MineLawHub - sandbox\backend\vocab_intent.pkl"
EMBED_DIM = 64
HIDDEN_DIM = 32
EPOCHS = 20
BATCH_SIZE = 4

class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = self.fc(embedded)
        x = self.relu(x)
        return self.out(x)

def tokenize(text):
    return text.lower().split()

def train():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        
    # Build Vocab
    all_words = []
    for item in data:
        all_words.extend(tokenize(item["text"]))
    
    word_counts = Counter(all_words)
    vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.items())}
    vocab["<UNK>"] = 0
    
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
        
    labels_map = {"static_law": 0, "general_query": 1}
    num_classes = len(labels_map)
    
    # Prepare Tensors
    text_data = []
    offsets = [0]
    labels = []
    
    for item in data:
        tokens = [vocab.get(w, 0) for w in tokenize(item["text"])]
        text_data.extend(tokens)
        offsets.append(len(tokens))
        labels.append(labels_map[item["label"]])
        
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_data = torch.tensor(text_data, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Model
    model = IntentClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Training Loop
    print("Training Intent Classifier...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(text_data, offsets)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), MODEL_PATH)
    print("Intent Classifier trained and saved!")

if __name__ == "__main__":
    train()
