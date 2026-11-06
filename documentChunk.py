import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# Path to the PDF you want to chat with (default: ./node.pdf)
path = os.getenv("PDF_PATH", "./node.pdf")
loader = PyPDFLoader(file_path=path)

docs = loader.load()

# print(docs[0].page_content)
# print(len(docs))
