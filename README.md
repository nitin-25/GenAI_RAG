# Document Q&A with RAG

A small Retrieval-Augmented Generation (RAG) app that answers questions from a PDF
and shows the pages each answer came from.

## How it works

1. **Load**: `PyPDFLoader` reads the PDF page by page (`documentChunk.py`)
2. **Chunk**: `RecursiveCharacterTextSplitter` (size 1000, overlap 200) (`chunks.py`)
3. **Embed and store**: OpenAI embeddings saved in ChromaDB (`embedd.py`)
4. **Retrieve**: top-3 similar chunks for the question (`retriever.py`)
5. **Answer**: a LangChain chain sends context and question to the LLM, answers only from the document, and returns page citations (`main.py`)

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your API key
```

Put a PDF at `./node.pdf` (or set `PDF_PATH` in `.env`), then:

```bash
python chunks.py     # optional: inspect the chunks
python embedd.py     # build the vector store
python main.py       # start chatting
```

## Roadmap

- [x] Basic RAG pipeline
- [x] Source citations (file and page)
- [ ] Chat history (follow-up questions)
- [ ] Evaluation set and chunk-size comparison
- [ ] FastAPI backend and React chat UI
