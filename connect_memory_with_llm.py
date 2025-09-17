import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

load_dotenv()



#  History helper 
history_store = [] 

def get_history(_):
    """Return previous Q/A pairs as string (or empty)."""
    if not history_store:
        return "No previous conversation."
    return "\n".join(
        [f"Q: {q}\nA: {a}" for q, a in history_store[-5:]]  # keep last 5
    )

def save_to_history(inp, out):
    history_store.append((inp["question"], out))
    return out

# HuggingFace LLm


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

CUSTOM_PROMPT = """
You are a helpful medical assistant.
Use the CONTEXT below and (if available) HISTORY of conversation
to answer the QUESTION.

CONTEXT:
{context}

HISTORY:
{history}

QUESTION:
{question}

Answer directly and concisely.
"""
prompt = PromptTemplate(
    template=CUSTOM_PROMPT,
    input_variables=["context", "history", "question"]
)

# Retriever
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Build chain with runnables 
qa_chain = (
    {
        "question": itemgetter("question"),
        "context": itemgetter("question") | retriever,  # take question -> docs
        "history": RunnableLambda(get_history),
    }
    | prompt
    | llm
)

# Run loop 
if __name__ == "__main__":
    while True:
        query = input("\nAsk something (or 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        result = qa_chain.invoke({"question": query})
        print("\nANSWER:\n", result.content)

        # save for next turn
        save_to_history({"question": query}, result.content)

        
