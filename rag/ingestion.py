"""Pinecone is a vector database used in AI and machine learning applications.

Why do we need it?

Large Language Models (LLMs) like OpenAI cannot remember huge amounts of custom data directly. To let an AI search through your documents, notes, PDFs, websites, or company knowledge, we:

Convert text into embeddings (vectors of numbers).
Store those vectors in a vector database like Pinecone.
When a user asks a question, convert the question into a vector.
Find the most similar vectors in the database.
Send the relevant information to the LLM to generate an answer.

This process is called RAG (Retrieval-Augmented Generation)."""
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("/Users/edenmarco/Desktop/langchain-course/mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))#type:ignore 

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")