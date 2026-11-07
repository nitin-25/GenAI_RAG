import os

import httpx
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

load_dotenv()

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

client = httpx.Client(verify=os.getenv("SSL_VERIFY", "true").lower() != "false")

# Initialize the embedding model (must match the one used in embedd.py)
#for openAI Mode
# embeddings = OpenAIEmbeddings(
#     model=os.getenv("EMBEDDING_MODEL"),
#     api_key=os.getenv("API_KEY"),
#     base_url=os.getenv("BASE_URL"),
#     http_client=client,
# )

#for ollama model
embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))


# Load existing Chroma DB from disk
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

if __name__ == "__main__":
    query = "what is node js"
    result = retriever.invoke(query)

    print(f"Retrieved {len(result)} chunks:")
    for i, doc in enumerate(result, start=1):
        print(f"\n-- Result {i} (page: {doc.metadata.get('page', 'N/A')}) ----")
        print(doc.page_content)
