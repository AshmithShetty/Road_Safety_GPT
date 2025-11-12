import pandas as pd
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
import logging
import sys

#CONFIGURATION
EXCEL_FILE_PATH = "GPT_Input_DB.xlsx"
DB_PERSIST_DIR = "./storage"  # Save the database
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5" #  embedding model
COLLECTION_NAME = "road_safety_db" # collection in ChromaDB


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def load_data_from_excel(file_path):
    """
    Loads data from the specified Excel file and converts it into
    LlamaIndex 'Document' objects with metadata.
    """
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

    documents = []
    for _, row in df.iterrows():
        
        main_text = str(row.get('data', ''))
        
       
        metadata = {
            "problem": str(row.get('problem', '')),
            "category": str(row.get('category', '')),
            "type": str(row.get('type', '')),
            "code": str(row.get('code', '')),
            "clause": str(row.get('clause', ''))
        }
        
        # Create a LlamaIndex Document
      
        doc = Document(text=main_text, metadata=metadata)
        documents.append(doc)
        
    print(f"Successfully loaded {len(documents)} documents from Excel.")
    return documents

def build_and_store_index(documents, persist_dir):
    """
    Builds the vector index from the documents and saves it to disk.
    """
    if not documents:
        print("No documents to index. Exiting.")
        return

    print("Initializing ChromaDB...")
    # Initialize the ChromaDB client
    
    db = chromadb.PersistentClient(path=persist_dir)
    
  
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    
    # Assign ChromaDB
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Define the storage context 
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print(f"Loading embedding model: {EMBED_MODEL_NAME}...")

   
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    
    # Configure LlamaIndex global settings
    Settings.embed_model = embed_model
    Settings.llm = None 
    Settings.chunk_size = 512 
    
    print("Embedding documents and building index... This may take a few minutes...")

    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    
    print("Persisting index to disk...")

    
    index.storage_context.persist(persist_dir=persist_dir)
    
    print(f"--- SUCCESS ---")
    print(f"Vector store created and saved at: {persist_dir}")

#MAIN EXECUTION
if __name__ == "__main__":
    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"Fatal Error: Input file not found at '{EXCEL_FILE_PATH}'")
        sys.exit(1)
        
    docs = load_data_from_excel(EXCEL_FILE_PATH)
    build_and_store_index(docs, DB_PERSIST_DIR)