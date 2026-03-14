"""
Detect and extract English-only pages from bilingual PDFs.
Uses language detection on extracted text to identify English pages.
"""

import os
import re
from pathlib import Path
import pdfplumber
from PyPDF2 import PdfReader, PdfWriter


def is_english_page(text: str) -> bool:
    """
    Determine if a page is primarily in English using heuristics.
    
    Args:
        text: Extracted text from a page
        
    Returns:
        True if page appears to be in English
    """
    if not text or len(text.strip()) < 50:
        return False
    
    # Count English words (simple heuristic: words with only ASCII letters)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    # Count non-ASCII characters (likely non-English)
    non_ascii = len([c for c in text if ord(c) > 127])
    
    # If more than 20% non-ASCII, likely not English
    if len(text) > 0 and (non_ascii / len(text)) > 0.2:
        return False
    
    # Check for common English legal terms
    english_terms = [
        'shall', 'thereof', 'herein', 'pursuant', 'regulation',
        'section', 'chapter', 'act', 'rule', 'provision'
    ]
    
    text_lower = text.lower()
    english_term_count = sum(1 for term in english_terms if term in text_lower)
    
    # If we have English words and some legal terms, likely English
    return len(words) > 10 and english_term_count > 0


def extract_english_pages(pdf_path: str, output_pdf_path: str, output_text_path: str) -> None:
    """
    Extract English-only pages from a bilingual PDF.
    
    Args:
        pdf_path: Path to input PDF
        output_pdf_path: Path to save English-only PDF
        output_text_path: Path to save extracted English text
    """
    try:
        english_pages = []
        english_text = []
        
        # First pass: identify English pages using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and is_english_page(text):
                    english_pages.append(page_num)
                    english_text.append(f"--- Page {page_num + 1} ---\n{text}\n\n")
        
        if not english_pages:
            print(f"✗ No English pages detected in {os.path.basename(pdf_path)}")
            return
        
        # Second pass: extract English pages to new PDF using PyPDF2
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        
        for page_num in english_pages:
            writer.add_page(reader.pages[page_num])
        
        # Save English-only PDF
        with open(output_pdf_path, 'wb') as f:
            writer.write(f)
        
        # Save English text
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(''.join(english_text))
        
        print(f"✓ Extracted {len(english_pages)} English pages from {os.path.basename(pdf_path)}")
        
    except Exception as e:
        print(f"✗ Error processing {os.path.basename(pdf_path)}: {str(e)}")


def main():
    """
    Main function to process bilingual PDFs and extract English pages.
    """
    # Define paths
    script_dir = Path(__file__).parent.parent
    pdf_dir = script_dir / "data" / "pdfs"
    english_pdf_dir = script_dir / "data" / "english_pdfs"
    english_text_dir = script_dir / "data" / "english_text"
    
    # Create output directories
    english_pdf_dir.mkdir(parents=True, exist_ok=True)
    english_text_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if PDF directory exists
    if not pdf_dir.exists():
        print(f"✗ PDF directory not found: {pdf_dir}")
        return
    
    # Get all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"✗ No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process for English extraction\n")
    
    # Process each PDF
    for pdf_file in pdf_files:
        output_pdf = english_pdf_dir / f"{pdf_file.stem}_english.pdf"
        output_text = english_text_dir / f"{pdf_file.stem}_english.txt"
        extract_english_pages(str(pdf_file), str(output_pdf), str(output_text))
    
    print(f"\n✓ English page extraction complete!")
    print(f"  PDFs saved to: {english_pdf_dir}")
    print(f"  Text saved to: {english_text_dir}")


if __name__ == "__main__":
    main()
