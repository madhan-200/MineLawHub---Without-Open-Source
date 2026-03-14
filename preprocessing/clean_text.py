"""
Clean and normalize extracted text from PDFs.
Removes headers, footers, page numbers, and normalizes whitespace.
"""

import os
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    # Remove page markers
    text = re.sub(r'--- Page \d+ ---', '', text)
    
    # Remove common headers/footers patterns
    # Remove lines with only page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove lines with only Roman numerals
    text = re.sub(r'^\s*[ivxlcdm]+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove common footer patterns (e.g., "Page 1 of 10")
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace while preserving paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing/leading whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Fix broken words at line endings (hyphenation)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Normalize multiple spaces to single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove lines that are likely headers/footers (very short lines at start/end)
    lines = text.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        # Skip very short lines (< 10 chars) that might be headers/footers
        # unless they're in the middle of the document
        if len(line.strip()) < 10 and (i < 3 or i > len(lines) - 3):
            continue
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Final cleanup: remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def process_file(input_path: str, output_path: str) -> None:
    """
    Process a single text file.
    
    Args:
        input_path: Path to input text file
        output_path: Path to save cleaned text
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        cleaned = clean_text(raw_text)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        print(f"✓ Cleaned {os.path.basename(input_path)}")
        
    except Exception as e:
        print(f"✗ Error processing {os.path.basename(input_path)}: {str(e)}")


def main():
    """
    Main function to clean all text files in data/text directory.
    """
    # Define paths
    script_dir = Path(__file__).parent.parent
    text_dir = script_dir / "data" / "text"
    cleaned_dir = script_dir / "data" / "cleaned"
    
    # Create output directory
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if text directory exists
    if not text_dir.exists():
        print(f"✗ Text directory not found: {text_dir}")
        print("Please run extract_text.py first.")
        return
    
    # Get all text files
    text_files = list(text_dir.glob("*.txt"))
    
    if not text_files:
        print(f"✗ No text files found in {text_dir}")
        return
    
    print(f"Found {len(text_files)} text file(s) to clean\n")
    
    # Process each file
    for text_file in text_files:
        output_file = cleaned_dir / text_file.name
        process_file(str(text_file), str(output_file))
    
    print(f"\n✓ Text cleaning complete! Files saved to {cleaned_dir}")


if __name__ == "__main__":
    main()
