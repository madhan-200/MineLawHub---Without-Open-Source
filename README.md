# MineLawHub — AI-Powered Mining Law Chatbot

> **100% Custom Transformer Architecture (2025)** | 19.6M Parameters | Zero External APIs | Fully Offline

MineLawHub is an AI-powered chatbot for Indian Mining Laws built entirely with **custom Transformer neural networks trained from scratch**. No OpenAI, no Gemini, no pre-trained models — everything runs locally on your machine.

---

## 🏗️ Architecture

| Component | Technology | Details |
|-----------|-----------|---------|
| **Tokenizer** | Custom BPE | 8,000 subword vocabulary trained on mining law corpus |
| **Encoder** | Transformer (RoPE + GQA + SwiGLU + RMSNorm) | 256-dim context-aware embeddings, 5M params |
| **Reranker** | Cross-Encoder | Query-document relevance scoring, 98.7% accuracy |
| **Decoder** | Transformer (Cross-Attention + Weight Tying) | Answer generation up to 200 tokens, 6M params |
| **Intent Classifier** | Transformer Encoder + Linear | Legal query vs greeting detection, 100% accuracy |
| **Search** | ChromaDB + Hybrid Search | Semantic + lexical + source-aware retrieval |
| **Backend** | FastAPI + PyTorch | REST API serving all models |
| **Frontend** | React + Material UI | Chatbot UI, Laws browser, About page |

---

## 📂 Project Structure

```
MineLawHub/
├── backend/
│   ├── main.py                          # FastAPI server
│   ├── custom_client.py                 # Transformer inference pipeline
│   ├── search_engine.py                 # ChromaDB hybrid search
│   ├── transformer_models.py            # All model architectures (PyTorch)
│   ├── train_bpe_tokenizer.py           # BPE tokenizer training + class
│   ├── train_transformer_encoder.py     # Encoder + Intent training
│   ├── train_reranker.py                # Cross-Encoder Reranker training
│   ├── train_decoder.py                 # Transformer Decoder training
│   └── bpe_tokenizer.json              # Trained tokenizer (generated)
├── frontend/
│   ├── src/
│   │   ├── pages/                       # HomePage, ChatPage, LawsPage, AboutPage
│   │   ├── components/                  # Navbar, Sidebar, ChatWindow, etc.
│   │   └── App.jsx                      # Main React app
│   └── package.json
├── data/
│   ├── text/                            # 6 source law documents (.txt)
│   └── training/                        # corpus.txt, intent_data.json, qa_data.json
├── embeddings/
│   └── rebuild_transformer_embeddings.py # ChromaDB rebuild script
├── preprocessing/
│   └── prepare_dataset.py               # Data preparation
├── requirements.txt                     # Python dependencies
├── .env.example                         # Environment config template
└── README.md                            # This file
```

---

## 🚀 How to Run (Step-by-Step)

### Prerequisites

- **Python 3.9+** (recommended: 3.10 or 3.11)
- **Node.js 16+** and **npm**
- **Git**
- **~4GB disk space** (for models + ChromaDB index)

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/madhan-200/MineLawHub---Without-Open-Source.git
cd MineLawHub---Without-Open-Source
```

### Step 2: Set Up Python Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train All Models (One-Time Setup)

These scripts train the custom Transformer models from scratch. Run them in order:

```bash
cd backend

# Step 4a: Train BPE Tokenizer (~30 seconds)
python train_bpe_tokenizer.py

# Step 4b: Train Transformer Encoder + Intent Classifier (~5-10 minutes)
python train_transformer_encoder.py

# Step 4c: Train Cross-Encoder Reranker (~3-5 minutes)
python train_reranker.py

# Step 4d: Train Transformer Decoder (~5-10 minutes)
python train_decoder.py
```

After training, you should have these files in `backend/`:
```
bpe_tokenizer.json          # BPE tokenizer
transformer_encoder.pth     # ~20 MB
transformer_intent.pth      # ~14 MB
reranker.pth                # ~20 MB
transformer_decoder.pth     # ~24 MB
```

### Step 5: Build ChromaDB Embeddings (One-Time Setup)

```bash
cd ../embeddings
python rebuild_transformer_embeddings.py
```

This creates `embeddings/chroma_store_v3/` with 6,281 indexed legal chunks.

### Step 6: Start the Backend Server

```bash
cd ../backend
python main.py
```

The API server starts at **http://localhost:8000**. You should see:
```
Loading Custom Transformer Models (2025 Architecture)...
  ✓ Transformer Encoder loaded (4,999,424 params)
  ✓ Intent Classifier loaded (3,556,994 params)
  ✓ Cross-Encoder Reranker loaded (5,032,449 params)
  ✓ Transformer Decoder loaded (6,049,024 params)
✓ All custom Transformer models loaded successfully
```

### Step 7: Install Frontend Dependencies

Open a **new terminal**:

```bash
cd frontend
npm install
```

### Step 8: Start the Frontend

```bash
npm start
```

The React app opens at **http://localhost:3000**.

---

## 🧪 Test Queries

Try these in the chatbot:

| Query | Expected Behavior |
|-------|-------------------|
| `What is the minimum age to work in mines?` | Section 40 — age limit |
| `Can women work underground?` | Section 46 — women employment |
| `What is mining act 1952?` | Act overview |
| `Duties of mine manager` | Section 17 |
| `What are penalties for violations?` | Sections 72A-74 |
| `Hello` | Greeting response |

---

## 📊 Model Training Results

| Model | Parameters | Training Result |
|-------|-----------|----------------|
| Transformer Encoder | 4,999,424 | Loss: 1.64 → 0.04 (30 epochs) |
| Intent Classifier | 3,556,994 | 100% accuracy (30 epochs) |
| Cross-Encoder Reranker | 5,032,449 | 98.7% accuracy (20 epochs) |
| Transformer Decoder | 6,049,024 | Loss: 8.1 → 1.96 (30 epochs) |
| **Total** | **19,637,891** | |

---

## 🔒 Privacy & Security

- ✅ **No external APIs** — Zero calls to OpenAI, Gemini, or any cloud AI
- ✅ **No pre-trained weights** — All models trained from scratch on our data
- ✅ **No internet required** — Everything runs locally after setup
- ✅ **No data leaves your machine** — All queries processed on-device

---

## 🛠️ Tech Stack

- **AI/ML**: PyTorch (custom Transformer models)
- **Backend**: FastAPI, ChromaDB, Python
- **Frontend**: React, Material UI, Framer Motion
- **Tokenization**: Custom BPE (Byte Pair Encoding)

---

## 📜 Mining Laws Covered

1. Mines Act, 1952
2. Mineral Conservation and Development Rules (MCDR), 2017
3. Mines and Minerals (Development and Regulation) Act (MMDR), 1957
4. Coal Mines Regulations, 2017
5. Mines Rules, 1955
6. Metalliferous Mines Regulations, 1961

---

## 👥 Team

| Name | Roll No | Contributions |
|------|---------|--------------|
| **Madhankumar S** | 7376222IT184 | Backend, Custom Transformer Models, Search Engine, Architecture |
| **Jayatchana Aravind M** | 7376222IT159 | React Frontend, Data Collection, Law Validation, Testing |

---

## 📝 License

This project is for educational purposes — built as part of an academic project.
