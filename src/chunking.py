def chunk_text(text, chunk_size=300, chunk_overlap=50):
    """
    Splits text into sliding windows of word tokens to preserve contextual integrity.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        # Stop if we reached the end of the text
        if end >= len(words):
            break
            
        start += (chunk_size - chunk_overlap)

    return chunks

def process_documents_into_chunks(documents, chunk_size=300, chunk_overlap=50):
    """
    Processes extracted pages into structured chunks with precise citation metadata.
    """
    structured_chunks = []
    chunk_id = 0

    for doc in documents:
        filename = doc["filename"]
        for page in doc["pages"]:
            page_num = page["page_number"]
            page_text = page["text"]

            text_chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            for sub_idx, chunk_content in enumerate(text_chunks):
                structured_chunks.append({
                    "chunk_id": chunk_id,
                    "filename": filename,
                    "page_number": page_num,
                    "chunk_index": sub_idx,
                    "text": chunk_content
                })
                chunk_id += 1

    return structured_chunks