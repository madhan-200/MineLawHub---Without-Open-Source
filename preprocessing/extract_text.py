"""
Extract text from PDF files in the data/pdfs directory.
Outputs individual text files for each PDF in data/text/.
"""

import os
import pdfplumber
from pathlib import Path


def extract_text_from_pdf(pdf_path: str, output_path: str) -> None:
    """
    Extract text from a single PDF file using pdfplumber.
    
    Args:
        pdf_path: Path to the input PDF file
        output_path: Path to save the extracted text
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_content = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    text_content.append(f"--- Page {page_num} ---\n")
                    text_content.append(text)
                    text_content.append("\n\n")
            
            # Write extracted text to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(''.join(text_content))
            
            print(f"✓ Extracted text from {os.path.basename(pdf_path)} ({len(pdf.pages)} pages)")
            
    except Exception as e:
        print(f"✗ Error extracting text from {os.path.basename(pdf_path)}: {str(e)}")


def main():
    """
    Main function to extract text from all PDFs in data/pdfs directory.
    """
    # Define paths
    script_dir = Path(__file__).parent.parent
    pdf_dir = script_dir / "data" / "pdfs"
    text_dir = script_dir / "data" / "text"
    
    # Create output directory if it doesn't exist
    text_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if PDF directory exists
    if not pdf_dir.exists():
        print(f"✗ PDF directory not found: {pdf_dir}")
        print("Please create the directory and add PDF files.")
        return
    
    # Get all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"✗ No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process\n")
    
    # Process each PDF
    for pdf_file in pdf_files:
        output_file = text_dir / f"{pdf_file.stem}.txt"
        extract_text_from_pdf(str(pdf_file), str(output_file))
    
    print(f"\n✓ Text extraction complete! Files saved to {text_dir}")


if __name__ == "__main__":
    main()
