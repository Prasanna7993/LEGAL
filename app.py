from pathlib import Path
from dotenv import load_dotenv

root = Path(__file__).resolve().parent
env_file = root / ".env"
if not env_file.exists():
    env_file = root / "myenv" / ".env"
load_dotenv(env_file)

from fastapi import FastAPI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# ----------------------------------
# Load Legal Documents
# ----------------------------------

loader = TextLoader("legal.txt")
documents = loader.load()

# ----------------------------------
# Split Documents
# ----------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# ----------------------------------
# Create Embeddings
# ----------------------------------

embeddings = OpenAIEmbeddings()

# ----------------------------------
# Create Vector Database
# ----------------------------------

vector_db = FAISS.from_documents(
    docs,
    embeddings
)

print("✅ Legal RAG Loaded Successfully")

# ----------------------------------
# Load LLM
# ----------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ----------------------------------
# FastAPI App
# ----------------------------------

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Legal RAG API Running"
    }

@app.get("/ask")
def ask_question(query: str):

    # Retrieve relevant chunks
    retrieved_docs = vector_db.similarity_search(
        query,
        k=2
    )

    context = "\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = f"""
You are a legal assistant.

Answer ONLY from the provided context.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return {
        "question": query,
        "answer": response.content
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )