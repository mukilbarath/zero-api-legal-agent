import os
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

warnings.filterwarnings("ignore")

def build_local_vector_db(pdf_path):
    print(f"Loading standard PDF: {pdf_path}...")
    
    # 1. Load the PDF
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

    # 2. Chunk the text
    print("Chunking document into manageable pieces...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # 500 characters per chunk
        chunk_overlap=50    # Slight overlap so clauses aren't cut in half
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Local Embeddings using Ollama
    print("Generating local embeddings via Ollama (Llama-3)...")
    # We use the same llama3 model you already downloaded to generate the vectors
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Store in local ChromaDB
    print("Storing chunks in local ChromaDB...")
    db_directory = "./chroma_db"
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=db_directory
    )
    
    print(f"\nSuccess! Vector database created at '{db_directory}'.")
    return vector_db

if __name__ == "__main__":
    # Ensure you have a simple PDF named sample_contract.pdf in your folder
    build_local_vector_db("sample_doc.pdf")