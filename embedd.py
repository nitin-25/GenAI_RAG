import os
import shutil

import httpx
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from chunks import chunks

load_dotenv()

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# Set SSL_VERIFY=false in .env only if your network requires it
client = httpx.Client(verify=os.getenv("SSL_VERIFY", "true").lower() != "false")

embedding = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    http_client=client,
)

if __name__ == "__main__":
    # Start fresh so re-running does not store duplicate chunks
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=PERSIST_DIR,  # saves to disk
    )
    print(f"Stored {len(chunks)} chunks in {PERSIST_DIR}")
