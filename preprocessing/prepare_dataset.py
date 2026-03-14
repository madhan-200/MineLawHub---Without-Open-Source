import os
import re
import json
import random
from pathlib import Path

def clean_text(text):
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i+chunk_size]))
    return chunks

def prepare_dataset():
    data_dir = Path(r"c:\MineLawHub - sandbox\data\cleaned")
    output_dir = Path(r"c:\MineLawHub - sandbox\data\training")
    output_dir.mkdir(exist_ok=True)
    
    all_chunks = []
    intent_data = []
    
    # Static Law Examples (from documents)
    for file_path in data_dir.glob("*.txt"):
        print(f"Processing {file_path.name}...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            cleaned = clean_text(text)
            chunks = get_chunks(cleaned)
            all_chunks.extend(chunks)
            
            # Label as static_law
            for chunk in chunks:
                if len(chunk) > 50:
                    intent_data.append({"text": chunk[:100], "label": "static_law"})

    # General Query Examples (Generated)
    general_queries = [
        "Hello", "Hi", "How are you?", "Who are you?", 
        "What can you do?", "Tell me a joke.", "What's the weather?",
        "Good morning", "Goodbye", "Thank you", "Help me",
        "What is your name?", "Are you a bot?", "Who made you?"
    ]
    for q in general_queries:
        intent_data.append({"text": q, "label": "general_query"})

    # Manual Augmentation with Ground Truth (Precision Booster)
    manual_qa = [
        {"question": "What is the minimum age for working in a mine?", "answer": "As per Section 40 of the Mines Act, no person below 18 years of age is allowed to work in any mine. Apprentices and trainees not below 16 years may work under supervision."},
        {"question": "Can children work in mines?", "answer": "No. Section 40 and 45 of the Mines Act 1952 strictly prohibit the employment and presence of anyone below 18 years of age in a mine."},
        {"question": "What are the mining age categories?", "answer": "The Mines Act defines 'adult' as a person who has completed 18 years. Apprentices must be at least 16 years old. No one under 18 is allowed for regular employment."},
        {"question": "Who is an adult under Mines Act?", "answer": "Under Section 2(b), an adult means a person who has completed his eighteenth year."},
        {"question": "Can women work below ground?", "answer": "Section 46 of the Mines Act prohibits the employment of women in any part of a mine which is below ground."},
        {"question": "What are the hours for women in mines?", "answer": "Women can only work above ground between 6 AM and 7 PM as per Section 46 of the Mines Act."}
    ]
    
    # Save Intent Data
    with open(output_dir / "intent_data.json", "w") as f:
        json.dump(intent_data, f, indent=2)
    
    # Save Corpus for Word2Vec
    with open(output_dir / "corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_chunks))
        
    # Generate Synthetic Q&A for Generator (Simple Section-based)
    qa_pairs = []
    for chunk in all_chunks:
        # Look for section numbers or titles
        match = re.search(r'(Section|Rule|Regulation)\s+(\d+[A-Z]?)', chunk, re.I)
        if match:
            entity_type = match.group(1)
            num = match.group(2)
            query = f"What is {entity_type} {num} about?"
            qa_pairs.append({"query": query, "context": chunk, "answer": chunk[:300] + "..."})
            
    with open(output_dir / "qa_data.json", "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Dataset prepared! \n- Chunks: {len(all_chunks)}\n- Intent samples: {len(intent_data)}\n- QA pairs: {len(qa_pairs)}")

if __name__ == "__main__":
    prepare_dataset()
