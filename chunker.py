from typing import List, Dict

def chunk_text(
        text: str,
        doc_name: str,
        page_number: int,
        chunk_size: int = 500,
        chunk_overlap: int = 50
) -> List[Dict]:
    """
    Slices text into overlapping chunks and attaches document metadata.
    
    Args:
        text: Raw text string from a document page
        doc_name: Name of the file (e.g., 'commercial_lease.pdf')
        page_number: The page number where this text originated
        chunk_size: Target character length of each chunk
        chunk_overlap: Character overlap between consecutive chunks
        
    Returns:
        List of dictionaries containing the chunk text and metadata dict.
    """
    # Clean up excess whitespace and blank lines
    cleaned_text = " ".join(text.split())

    if not cleaned_text:
        return[]

    chunks = []
    start = 0
    chunk_index = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = start + chunk_size
        chunk_content = cleaned_text[start:end]

        # Build unique metadata tag for this specific piece of text
        chunk_record = {
            "text": chunk_content,
            "metadata": {
                "doc_name": doc_name,
                "page": page_number,
                "chunk_id": f"{doc_name}_p{page_number}_{chunk_index}"
            }
        }
        chunks.append(chunk_record)

        # Move the sliding window forward by (chunk_size - chunk_overlap)
        start += (chunk_size - chunk_overlap)
        chunk_index += 1

    return chunks
if __name__ == "__main__":
    # Smoke Test: Test chunking with a simulated legal clause
    sample_legal_text = (
        "SECTION 8.2 - EARLY TERMINATION PENALTIES. In the event that the Tenant "
        "vacates the premises or terminates this Commercial Agreement prior to the expiration "
        "of the initial twenty-four (24) month lease term, the Tenant shall immediately forfeit "
        "the Security Deposit amounting to $10,000 USD. Furthermore, the Tenant agrees to pay "
        "a liquidated damages fee equivalent to two (2) months of standard rent within thirty (30) "
        "calendar days of delivering written notice to the Landlord. Failure to remit this payment "
        "shall incur statutory interest at a rate of 8.5% per annum."
    )
    print("--- Testing chunk_text function ---")
    # We use small chunk_size (150 chars) and 30-char overlap so we can see the slicing clearly
    sample_chunks = chunk_text(
        text=sample_legal_text,
        doc_name="lease_contract.pdf",
        page_number=4,
        chunk_size=150,
        chunk_overlap=30
    )

    print(f"Generated {len(sample_chunks)} chunks from sample text.\n")
    for i, c in enumerate(sample_chunks):
        print(f"Chunk #{i + 1} | ID: {c['metadata']['chunk_id']}")
        print(f"Text: \"{c['text']}\"")
        print(f"Length: {len(c['text'])} chars")
        print("-" * 50)
              