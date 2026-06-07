from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

print("loading document...")

loader = TextLoader("rag/mediumblog1.txt", encoding="utf-8")
docs = loader.load()

print("splitting document...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print(f"created {len(chunks)} chunks")

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

print("creating faiss index...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local("faiss_index")

print("done")