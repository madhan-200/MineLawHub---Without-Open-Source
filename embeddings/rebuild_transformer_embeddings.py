"""
Rebuild ChromaDB embeddings using Transformer Encoder (256-dim).
Replaces the old Word2Vec-based embeddings (128-dim).

Creates: chroma_store_v3/
"""

import os
import sys
import re
from pathlib import Path

# Add backend to path for imports
BACKEND_DIR = str(Path(__file__).parent.parent / "backend")
sys.path.insert(0, BACKEND_DIR)

import chromadb
from custom_client import CustomClient

# ─── Configuration ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "text"
CHROMA_PATH = str(PROJECT_ROOT / "embeddings" / "chroma_store_v3")
COLLECTION_NAME = "mining_law_docs"

# Chunk size for splitting documents
CHUNK_SIZE = 500       # Characters per chunk
CHUNK_OVERLAP = 50     # Overlap between chunks


def clean_text(text):
    """Remove garbled OCR artifacts."""
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def detect_section(text):
    """Try to detect section/regulation number from chunk text."""
    # Look for patterns like "Section 40", "40.", "Regulation 27"
    match = re.search(r'(?:Section|Regulation|Rule)\s+(\d+[A-Z]?)', text[:200], re.IGNORECASE)
    if match:
        return f"section {match.group(1)}"
    
    # Check if text starts with a number followed by period
    match = re.search(r'^(\d+[A-Z]?)\.', text.strip())
    if match:
        return f"section {match.group(1)}"
    
    return "general"


def build():
    print("=" * 60)
    print("Rebuilding ChromaDB with Transformer Embeddings (256-dim)")
    print("=" * 60)
    
    # Initialize custom client (loads Transformer models)
    print("\n[1/4] Loading Transformer models...")
    client = CustomClient()
    
    # Find all text files in data/raw
    print("\n[2/4] Loading source documents...")
    if not DATA_DIR.exists():
        print(f"  ⚠ Data directory not found: {DATA_DIR}")
        print("  Trying alternative paths...")
        # Try finding text files elsewhere
        alt_paths = [
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "data" / "training",
        ]
        found_files = []
        for path in alt_paths:
            if path.exists():
                found_files.extend(path.glob("*.txt"))
        if not found_files:
            print("  ERROR: No source documents found!")
            return
    else:
        found_files = list(DATA_DIR.glob("*.txt")) + list(DATA_DIR.glob("*.pdf"))
    
    # Also check for the corpus file as a source
    corpus_path = PROJECT_ROOT / "data" / "training" / "corpus.txt"
    if corpus_path.exists() and corpus_path not in found_files:
        found_files.append(corpus_path)
    
    print(f"  Found {len(found_files)} source files")
    
    # Process all documents into chunks
    print("\n[3/4] Chunking documents...")
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for file_path in found_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Clean the text
            cleaned = clean_text(text)
            if len(cleaned) < 50:
                continue
            
            # Get source name from filename
            source_name = file_path.stem
            
            # Chunk the text
            chunks = chunk_text(cleaned)
            
            for i, chunk in enumerate(chunks):
                # Skip very short or garbled chunks
                if len(chunk) < 30:
                    continue
                non_ascii = sum(1 for ch in chunk[:200] if ord(ch) > 127)
                if non_ascii > len(chunk[:200]) * 0.3:
                    continue
                
                section = detect_section(chunk)
                doc_id = f"{source_name}_chunk_{i}"
                
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source_file": source_name,
                    "section": section,
                    "chunk_index": i
                })
                all_ids.append(doc_id)
        
        except Exception as e:
            print(f"  ⚠ Error processing {file_path.name}: {e}")
    
    print(f"  Total chunks: {len(all_chunks)}")
    
    if not all_chunks:
        print("  ERROR: No chunks to index!")
        return
    
    # Generate embeddings and store in ChromaDB
    print("\n[4/4] Generating Transformer embeddings and building index...")
    
    # Remove old v3 store if exists
    import shutil
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Process in batches
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        batch_chunks = all_chunks[i:batch_end]
        batch_metas = all_metadatas[i:batch_end]
        batch_ids = all_ids[i:batch_end]
        
        # Generate embeddings with Transformer
        batch_embeddings = []
        for chunk in batch_chunks:
            emb = client.get_embedding(chunk)
            batch_embeddings.append(emb)
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_chunks,
            metadatas=batch_metas
        )
        
        if (i + batch_size) % 200 == 0 or i == 0:
            print(f"  Indexed {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks...")
    
    print(f"\n✓ ChromaDB v3 built at {CHROMA_PATH}")
    print(f"  Documents: {collection.count()}")
    print(f"  Embedding dim: 256 (Transformer)")


if __name__ == "__main__":
    build()
    print("\n✓ ChromaDB rebuild complete!")
