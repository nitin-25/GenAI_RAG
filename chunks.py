from documentChunk import docs
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)

chunks = text_splitter.split_documents(docs)

if __name__ == "__main__":
    # Run `python chunks.py` to inspect how the PDF was split
    print("Pages loaded:", len(docs))
    print("Total chunks:", len(chunks))

    sizes = [len(c.page_content) for c in chunks]
    print("Chunk size -> min:", min(sizes), "max:", max(sizes), "avg:", sum(sizes) // len(sizes))

    for i, c in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i} | page {c.metadata.get('page')} | {len(c.page_content)} chars ---")
        print(c.page_content)
