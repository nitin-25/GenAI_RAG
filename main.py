import os

import httpx
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from retriever import retriever

load_dotenv()

client = httpx.Client(verify=os.getenv("SSL_VERIFY", "true").lower() != "false")

llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
    http_client=client,
)

# NOTE: reconstructed from a screenshot where some lines were cut off.
# Compare with your original prompt and restore the exact wording if needed.
template = """You are an intelligent AI assistant designed to answer questions using the provided context.

Your task is to generate accurate, relevant, and well-explained answers based strictly on the information in the context.

Instructions:
Use the provided context as the primary and authoritative source for answering the question.
Understand the context and generate a clear, comprehensive, and meaningful answer rather than simply copying text verbatim.
You may rephrase, summarize, simplify, reorganize, and elaborate on the information to make the answer easier to understand.
Explain technical concepts, processes, and terminology in a beginner-friendly manner whenever appropriate.
You may use examples, analogies, or step-by-step explanations only when they are consistent with the information in the context.
Do not introduce external facts, assumptions, personal opinions, or information that is not supported by the context.
If the context partially answers the question, provide the available information and clearly mention what is missing.
If the answer cannot be found in the context, respond with: "I couldn't find sufficient information in the provided document."
Avoid repeating information unnecessarily.
Structure the answer using headings, bullet points, or numbered steps wherever they improve readability.

Context:

{context}

Question:

{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Connect everything for RAG, now returning the answer AND its sources (citations)
rag_chain = (
    RunnableParallel(docs=retriever, question=RunnablePassthrough())
    | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
    | RunnablePassthrough.assign(answer=prompt | llm | StrOutputParser())
    | (
        lambda x: {
            "answer": x["answer"],
            "sources": sorted(
                {
                    (
                        os.path.basename(d.metadata.get("source", "unknown")),
                        d.metadata.get("page", 0) + 1,  # PyPDFLoader pages start at 0
                    )
                    for d in x["docs"]
                }
            ),
        }
    )
)

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("RAG assistant is ready. Type exit or quit to end.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_query = input("Enter a question: ").strip()

            if user_query.lower() in ["exit", "quit", "q"]:
                print("Exiting...")
                break

            if not user_query:
                continue

            print("Searching and generating answer...\n")
            result = rag_chain.invoke(user_query)

            print(f"Answer:\n{result['answer']}\n")
            print("Sources:")
            for filename, page in result["sources"]:
                print(f"  - {filename}, page {page}")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("Session interrupted! Exiting...")
            break
        except Exception as e:
            print(f"Something went wrong: {e}\n")
