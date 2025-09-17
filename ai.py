import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know, say "I don't know".
Keep the answer concise (max 3 sentences).

Question: {question}
Context: {context}
Answer:
"""

#  Embeddings & vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = FAISS.from_texts([""], embeddings)  

#  LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

#  Loader 
def load_page(url: str):
    loader = WebBaseLoader(url)
    return loader.load()

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def index_docs(docs):
    vector_store.add_documents(docs)

def retrieve_docs(query: str):
    return vector_store.similarity_search(query, k=4)

def answer_question(question: str, context: str):
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    chain = prompt | llm
    return chain.invoke({"question": question, "context": context})

if __name__ == "__main__":
    url = input("Enter URL: ")
    docs = load_page(url)
    chunks = split_text(docs)
    index_docs(chunks)

    while True:
        q = input("Ask a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        retrieved = retrieve_docs(q)
        ctx = "\n\n".join([doc.page_content for doc in retrieved])
        ans = answer_question(q, ctx)
        print("Answer:", ans.content)
