import os
import sys
from pathlib import Path

# Add backend to path to import CustomClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from custom_client import CustomClient
import chromadb
import re
from typing import List, Dict, Tuple

# ============================================================
# FIX #1: FORM/TEMPLATE FILTER
# Detects and filters out blank-form/template/schedule chunks
# that pollute search results with irrelevant content.
# ============================================================
FORM_INDICATORS = [
    r'^FORM\s*[–\-—]\s*[A-Z]',       # "FORM – O", "FORM-C"
    r'^\d*\s*\[?FORM\s*[–\-—]\s*[A-Z]', # "1[FORM – J]"
    r'Name of Mine[…\.]{3,}',          # "Name of Mine………"
    r'State[…\.]{3,}District',         # "State………District"
    r'SN\s+Nam\w*\s+Na',              # Table headers in forms
    r'S\.?\s*No\.?\s+Name\s+of',       # "S.No Name of"
    r'SCHEDULE\s*$',                   # Standalone "SCHEDULE"
    r'^\s*ANNEXURE',                   # Annexure headers
]

def is_form_chunk(text: str) -> bool:
    """Check if a chunk is primarily a form template (not substantive legal text)."""
    text_trimmed = text.strip()[:500]
    
    # Check for form indicators
    for pattern in FORM_INDICATORS:
        if re.search(pattern, text_trimmed, re.IGNORECASE | re.MULTILINE):
            # Confirm it's a form by checking for fill-in-the-blank patterns
            blank_count = len(re.findall(r'[…\.]{3,}|_{3,}', text_trimmed))
            if blank_count >= 2:
                return True
    
    # Check for very high density of blanks/dots (form templates)
    blank_count = len(re.findall(r'[…\.]{4,}|_{4,}', text))
    if blank_count > 5 and len(text.split()) < 200:
        return True
    
    return False


# ============================================================
# FIX #2: IMPROVED SECTION METADATA EXTRACTION
# Uses multiple regex patterns to robustly detect section numbers.
# ============================================================
def extract_section_info(text: str) -> str:
    """
    Extract the most specific section/rule/regulation number from chunk text.
    Tries multiple patterns to handle various formatting styles.
    """
    text_start = text[:600]  # Look at beginning
    
    # Pattern 1: "40. Employment of persons..." (number-dot at start of provision)
    match = re.search(r'(?:^|\n)\s*(\d+[A-Z]?)\.\s+[A-Z]', text_start)
    if match:
        return f"section {match.group(1)}"
    
    # Pattern 2: "Section 40", "Rule 29B", "Regulation 64", "Chapter V"
    match = re.search(r'(Section|Rule|Regulation|Chapter|Part)\s+(\d+[A-Z]?(?:\s*\(\d+\))?)', text_start, re.IGNORECASE)
    if match:
        return f"{match.group(1).lower()} {match.group(2)}"
    
    # Pattern 3: "[See Rule 29F(2)]" or "(See rule 29B)"
    match = re.search(r'\[?[Ss]ee\s+(Rule|Section|Regulation)\s+(\d+[A-Z]?)', text_start)
    if match:
        return f"{match.group(1).lower()} {match.group(2)}"
    
    # Pattern 4: "rule 29B" in lowercase
    match = re.search(r'\brule\s+(\d+[A-Z]?)\b', text_start, re.IGNORECASE)
    if match:
        return f"rule {match.group(1)}"
    
    # Pattern 5: "section 40" or "sub-section(2) of section 40" anywhere in text
    match = re.search(r'\bsection\s+(\d+[A-Z]?)\b', text_start, re.IGNORECASE)
    if match:
        return f"section {match.group(1)}"
    
    return ""


# ============================================================
# FIX #3: IMPROVED LEGAL-BOUNDARY CHUNKING
# Better splitting that respects provision boundaries and
# handles numbered sections (e.g., "40. Employment of...")
# ============================================================
def chunk_text(text: str, chunk_size: int = 400) -> List[str]:
    """
    Legal-boundary aware chunking with improved splitting.
    Prioritizes splitting at:
    1. Major section headers (Section 40, Rule 12, Chapter V)
    2. Numbered provision starts (40. Employment of...)
    3. Paragraph boundaries
    """
    # Split by major legal headers AND numbered provisions
    # Use a lookahead-based split so the header stays attached to its content
    header_pattern = r'(?=\n\s*(?:Section|Rule|Regulation|Chapter|Part)\s+\d+[A-Z]?[\s\.\-:])|(?=\n\s*\d+[A-Z]?\.\s+[A-Z][a-z])'
    
    parts = re.split(header_pattern, text)
    
    # Each part now starts with its header (section number). Build chunks.
    chunks = []
    current_chunk = ""
    
    for part in parts:
        if len(current_chunk) + len(part) < chunk_size * 6:
            current_chunk += part
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = part
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    # Final pass: split oversized chunks by paragraphs
    final_chunks = []
    for chunk in chunks:
        if len(chunk.split()) > chunk_size * 1.5:
            paras = chunk.split('\n\n')
            sub_chunk = ""
            for p in paras:
                if len(sub_chunk.split()) + len(p.split()) < chunk_size:
                    sub_chunk += "\n\n" + p
                else:
                    if sub_chunk.strip():
                        final_chunks.append(sub_chunk.strip())
                    sub_chunk = p
            if sub_chunk.strip():
                final_chunks.append(sub_chunk.strip())
        else:
            final_chunks.append(chunk)
            
    # Filter: minimum 50 chars, and not a form template
    return [c for c in final_chunks if len(c.strip()) > 50]


def rebuild_embeddings():
    script_dir = Path(__file__).parent.parent
    cleaned_dir = script_dir / "data" / "cleaned"
    chroma_dir = script_dir / "embeddings" / "chroma_store_v2"
    
    print("=" * 60)
    print("REBUILD CUSTOM EMBEDDINGS v2 - Root Cause Fix Edition")
    print("=" * 60)
    
    print("\nInitializing Custom Client for Embeddings...")
    client_custom = CustomClient()
    
    print("Connecting to ChromaDB...")
    client_chroma = chromadb.PersistentClient(path=str(chroma_dir))
    
    collection_name = "mining_law_docs"
    try:
        client_chroma.delete_collection(name=collection_name)
        print("  Deleted old collection.")
    except:
        pass
    
    collection = client_chroma.create_collection(name=collection_name)
    
    text_files = list(cleaned_dir.glob("*.txt"))
    all_docs = []
    form_filtered = 0
    short_filtered = 0
    
    for text_file in text_files:
        source_name = text_file.stem
        print(f"\nProcessing {source_name}...")
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = chunk_text(text)
        kept = 0
        for i, chunk in enumerate(chunks):
            # FIX #1: Filter form templates
            if is_form_chunk(chunk):
                form_filtered += 1
                print(f"  [FILTERED FORM] chunk {i}: {chunk[:60]}...")
                continue
            
            # FIX #2: Better section extraction
            section = extract_section_info(chunk)
            
            all_docs.append({
                'id': f"{source_name}_chunk_{i}",
                'text': chunk,
                'metadata': {'source_file': source_name, 'section': section if section else 'N/A'}
            })
            kept += 1
        
        print(f"  → {kept} chunks kept, {len(chunks) - kept} filtered")

    print(f"\n{'=' * 60}")
    print(f"Total chunks to index: {len(all_docs)}")
    print(f"Form templates filtered: {form_filtered}")
    print(f"{'=' * 60}")
    
    print(f"\nGenerating custom embeddings for {len(all_docs)} chunks...")
    
    embeddings = []
    texts = []
    ids = []
    metadatas = []
    
    for doc in all_docs:
        emb = client_custom.get_embedding(doc['text'])
        embeddings.append(emb)
        texts.append(doc['text'])
        ids.append(doc['id'])
        metadatas.append(doc['metadata'])
        
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    # Print section coverage report
    print(f"\n{'=' * 60}")
    print("SECTION COVERAGE REPORT")
    print(f"{'=' * 60}")
    
    sources = {}
    na_chunks = []
    for doc in all_docs:
        src = doc['metadata']['source_file']
        sec = doc['metadata']['section']
        if src not in sources:
            sources[src] = {'total': 0, 'with_section': 0, 'sections': set()}
        sources[src]['total'] += 1
        if sec != 'N/A':
            sources[src]['with_section'] += 1
            sources[src]['sections'].add(sec)
        else:
            na_chunks.append(f"  {doc['id']}: {doc['text'][:80]}...")
    
    for src, info in sorted(sources.items()):
        pct = (info['with_section'] / info['total'] * 100) if info['total'] > 0 else 0
        print(f"\n{src}: {info['total']} chunks, {info['with_section']} with section ({pct:.0f}%)")
        if info['sections']:
            print(f"  Sections: {', '.join(sorted(info['sections'], key=lambda x: x.split()[-1] if x.split() else x))}")
    
    if na_chunks:
        print(f"\nChunks with N/A section ({len(na_chunks)}):")
        for c in na_chunks[:10]:
            print(c)
        if len(na_chunks) > 10:
            print(f"  ... and {len(na_chunks) - 10} more")
    
    print(f"\n✓ Successfully re-indexed {len(all_docs)} chunks with custom embeddings!")

if __name__ == "__main__":
    rebuild_embeddings()
