import torch
import torch.nn as nn
import torch.optim as optim
import json
import pickle
import os
import random

# Configuration
QA_DATA_PATH = r"c:\MineLawHub - sandbox\data\training\qa_data.json"
MODEL_PATH = r"c:\MineLawHub - sandbox\backend\generator.pth"
VOCAB_PATH = r"c:\MineLawHub - sandbox\backend\vocab_gen.pkl"
EMBED_DIM = 128
HIDDEN_DIM = 256
MAX_LEN = 50
EPOCHS = 50
BATCH_SIZE = 16

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        embedded = self.embedding(x).unsqueeze(1)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.out(output.squeeze(1))
        return prediction, hidden

def tokenize(text):
    return text.lower().replace('.', ' . ').replace(',', ' , ').split()[:MAX_LEN]

def train():
    with open(QA_DATA_PATH, "r") as f:
        data = json.load(f)

    # Build Vocab
    all_tokens = []
    for item in data:
        all_tokens.extend(tokenize(item["query"]))
        all_tokens.extend(tokenize(item["answer"]))
    
    unique_tokens = sorted(list(set(all_tokens)))
    vocab = {tok: i+4 for i, tok in enumerate(unique_tokens)}
    vocab["<PAD>"] = 0
    vocab["<SOS>"] = 1
    vocab["<EOS>"] = 2
    vocab["<UNK>"] = 3
    
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)

    # Prepare Data
    input_tensors = []
    target_tensors = []
    
    for item in data:
        in_tokens = [vocab.get(t, 3) for t in tokenize(item["query"])]
        out_tokens = [vocab.get(t, 3) for t in tokenize(item["answer"])]
        
        # Padding
        in_tokens = (in_tokens + [0] * MAX_LEN)[:MAX_LEN]
        out_tokens = ([1] + out_tokens + [2] + [0] * MAX_LEN)[:MAX_LEN]
        
        input_tensors.append(in_tokens)
        target_tensors.append(out_tokens)
        
    input_tensors = torch.tensor(input_tensors, dtype=torch.long)
    target_tensors = torch.tensor(target_tensors, dtype=torch.long)

    # Model
    encoder = Encoder(len(vocab), EMBED_DIM, HIDDEN_DIM)
    decoder = Decoder(len(vocab), EMBED_DIM, HIDDEN_DIM)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("Training Seq2Seq Generator...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for i in range(0, len(input_tensors), BATCH_SIZE):
            batch_input = input_tensors[i:i+BATCH_SIZE]
            batch_target = target_tensors[i:i+BATCH_SIZE]
            
            optimizer.zero_grad()
            hidden = encoder(batch_input)
            
            loss = 0
            decoder_input = torch.tensor([1] * batch_input.size(0), dtype=torch.long)
            for t in range(1, batch_target.size(1)):
                output, hidden = decoder(decoder_input, hidden)
                loss += criterion(output, batch_target[:, t])
                decoder_input = batch_target[:, t] # Teacher forcing
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(input_tensors):.4f}")

    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict()
    }, MODEL_PATH)
    print("Generator trained and saved!")

if __name__ == "__main__":
    train()
