"""
Build embeddings from cleaned text and store in ChromaDB.
Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings.
"""

import os
import re
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Approximate number of words per chunk
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Only keep substantial chunks
            chunks.append(chunk)
    
    return chunks


def extract_section_info(text: str) -> str:
    """
    Extract section or chapter information from text.
    
    Args:
        text: Text chunk
        
    Returns:
        Section identifier or empty string
    """
    # Look for section patterns
    section_match = re.search(r'(Section|Chapter|Rule|Regulation)\s+\d+[A-Z]?', text, re.IGNORECASE)
    if section_match:
        return section_match.group(0)
    
    return ""


def process_text_file(file_path: str, source_name: str) -> List[Dict]:
    """
    Process a text file and create chunks with metadata.
    
    Args:
        file_path: Path to text file
        source_name: Name of source document
        
    Returns:
        List of dictionaries containing chunks and metadata
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = chunk_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            section = extract_section_info(chunk)
            
            doc = {
                'id': f"{source_name}_chunk_{i}",
                'text': chunk,
                'metadata': {
                    'source_file': source_name,
                    'chunk_index': i,
                    'section': section if section else 'N/A'
                }
            }
            documents.append(doc)
        
        return documents
        
    except Exception as e:
        print(f"✗ Error processing {source_name}: {str(e)}")
        return []


def build_embeddings():
    """
    Main function to build embeddings and store in ChromaDB.
    """
    # Define paths
    script_dir = Path(__file__).parent.parent
    cleaned_dir = script_dir / "data" / "cleaned"
    chroma_dir = script_dir / "embeddings" / "chroma_store"
    
    # Create ChromaDB directory
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if cleaned text directory exists
    if not cleaned_dir.exists():
        print(f"✗ Cleaned text directory not found: {cleaned_dir}")
        print("Please run clean_text.py first.")
        return
    
    # Get all cleaned text files
    text_files = list(cleaned_dir.glob("*.txt"))
    
    if not text_files:
        print(f"✗ No text files found in {cleaned_dir}")
        return
    
    print(f"Found {len(text_files)} text file(s) to process\n")
    
    # Initialize sentence transformer model
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB client
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    
    # Create or get collection
    collection_name = "mining_law_docs"
    
    # Delete existing collection if it exists (for fresh build)
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Mining Acts, Rules, and Regulations"}
    )
    
    # Process all files
    all_documents = []
    
    for text_file in text_files:
        source_name = text_file.stem
        print(f"Processing {source_name}...")
        documents = process_text_file(str(text_file), source_name)
        all_documents.extend(documents)
    
    if not all_documents:
        print("✗ No documents to embed")
        return
    
    print(f"\nGenerating embeddings for {len(all_documents)} chunks...")
    
    # Extract texts for embedding
    texts = [doc['text'] for doc in all_documents]
    ids = [doc['id'] for doc in all_documents]
    metadatas = [doc['metadata'] for doc in all_documents]
    
    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Add to ChromaDB
    print("Storing embeddings in ChromaDB...")
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"\n✓ Successfully created {len(all_documents)} embeddings!")
    print(f"✓ ChromaDB collection '{collection_name}' saved to {chroma_dir}")
    
    # Verify
    count = collection.count()
    print(f"✓ Verified: Collection contains {count} documents")


if __name__ == "__main__":
    build_embeddings()
