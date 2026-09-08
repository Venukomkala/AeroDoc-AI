import os
import pymupdf as fitz

def extract_text_from_pdf(pdf_path):
    """
    Extracts raw text page-by-page from a single PDF file.
    Retains page number metadata for accurate document citation.
    """
    document = fitz.open(pdf_path)
    pages = []

    for page_number in range(len(document)):
        page = document[page_number]
        text = page.get_text("text")

        # Skip empty or whitespace-only pages
        if text and text.strip():
            pages.append({
                "page_number": page_number + 1,
                "text": text.strip()
            })

    document.close()
    return pages

def load_all_raw_documents(data_folder="data/raw"):
    """
    Loads all PDF documents from the specified raw data directory.
    Handles relative and absolute path resolution.
    """
    # Normalize path relative to project root if needed
    if not os.path.isabs(data_folder):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        target_folder = os.path.join(base_dir, data_folder)
    else:
        target_folder = data_folder

    if not os.path.exists(target_folder):
        raise FileNotFoundError(f"Directory '{target_folder}' does not exist.")

    pdf_files = [f for f in os.listdir(target_folder) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"[Ingestion Warning] No PDF files found in {target_folder}")
        return []

    all_docs = []
    for filename in pdf_files:
        filepath = os.path.join(target_folder, filename)
        print(f"[Ingestion] Extracting: {filename}...")
        pages = extract_text_from_pdf(filepath)
        
        all_docs.append({
            "filename": filename,
            "total_pages": len(pages),
            "pages": pages
        })

    return all_docs

# Backwards compatibility alias
process_all_pdfs = load_all_raw_documents