import os
import chromadb
import logging
import sys
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    QueryBundle
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

#CONFIGURATION 
DB_PERSIST_DIR = "./storage"  # Saved DB 
COLLECTION_NAME = "road_safety_db" # collection name 
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5" # Embedding model 
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3" #  reranker
LOCAL_LLM_MODEL = "llama3:8b" # Ollama model 
CLOUD_LLM_MODEL = "models/gemini-1.5-pro" # Google Gemini Pro model

# Configure logging 
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def create_query_engine(llm_to_use):
    """
    Builds and returns the complete RAG query engine.
    """
    print("--- Initializing Query Engine ---")

    # CONFIGURE GLOBAL SETTINGS
    # Set the embedding model
    
    print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    
    # We set the LLM. This will be either Ollama or Gemini.
    Settings.llm = llm_to_use
    print(f"LLM set to: {llm_to_use.metadata.model_name}")

    #  LOAD THE VECTOR DATABASE
    print(f"Loading vector database from: {DB_PERSIST_DIR}")
    # Initialize ChromaDB client
    db = chromadb.PersistentClient(path=DB_PERSIST_DIR)
    
    # Get specific collection
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    
    # Assign ChromaDB as vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load the index from existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )
    print("Successfully loaded index from vector database.")

    #CONFIGURE THE RERANKER
    # This model runs locally.
    print(f"Loading reranker model: {RERANKER_MODEL_NAME}...")
    # Takes the top 10 search results
  
    reranker = FlagEmbeddingReranker(
        model=RERANKER_MODEL_NAME,
        top_n=5,  # Retrieve 10, return 5
        use_fp16=True 
    )
    print("Reranker loaded.")

    # QUERY ENGINE 
   
    query_engine = index.as_query_engine(
        similarity_top_k=10,  # Retrieve the top 10 most similar results
        node_postprocessors=[reranker] # Rerank those 10 to get the best 5
    )
    print("--- Query Engine is Ready ---")
    return query_engine

def get_llm(google_api_key=None):
    """
    Returns the appropriate LLM instance based on whether an
    API key is provided (Cloud mode vs Local mode).
    """
    if google_api_key:
        print("Google API Key found. Using CLOUD (Gemini 1.5 Pro) LLM.")
        os.environ["GOOGLE_API_KEY"] = google_api_key
        return GoogleGenAI(model_name=CLOUD_LLM_MODEL)
    else:
        print("No Google API Key found. Using LOCAL (Ollama Llama 3) LLM.")
        try:
            return Ollama(model=LOCAL_LLM_MODEL, request_timeout=120.0)
        except Exception as e:
            print(f"\n--- ERROR ---")
            print(f"Failed to connect to Ollama: {e}")
            print("Please make sure Ollama is installed and running.")
            print("You can download it from https://ollama.com")
            print("After installing, run 'ollama pull llama3:8b' in your terminal.")
            print("-------------")
            sys.exit(1) # Exit the script if can't connect

#TESTING
# We can run this file directly to test if our engine works.
if __name__ == "__main__":
    # Test LOCAL (Ollama) setup
    print("Testing LOCAL (Ollama) query engine...")
    local_llm = get_llm()
    query_engine = create_query_engine(local_llm)
    
    print("\n--- Test Query ---")
    print("Query: 'What are the rules for a STOP sign?'")
    
    # Format prompt to get better cited answers
    query_str = """
    Query: 'What are the rules for a STOP sign?'
    
    Please answer the query based *only* on the provided context.
    For your answer, please provide:
    1.  The direct answer to the query.
    2.  The source 'code' (e.g., IRC:67-2022).
    3.  The source 'clause' (e.g., 14.4).
    """
    
    response = query_engine.query(query_str)
    
    print("\n--- Test Response ---")
    print(response)
    print("\n--- Source Nodes ---")
    for node in response.source_nodes:
        print(f"Score: {node.score:.4f}")
        print(f"Metadata: {node.node.metadata}")
        print(f"Text: {node.node.get_content()[:100]}...")
        print("---")