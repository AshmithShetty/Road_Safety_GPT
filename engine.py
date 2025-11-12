import os
import chromadb
import logging
import sys
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    QueryBundle,
    PromptTemplate
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI


#CONFIGURATION 
DB_PERSIST_DIR = "./storage"
COLLECTION_NAME = "road_safety_db"
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"

LOCAL_LLM_MODEL = "llama3:8b"
CLOUD_LLM_MODEL = "models/gemini-1.5-pro"

# Configure logging 
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



PROMPT_TEMPLATE = """
You are a specialized AI assistant for the National Road Safety Hackathon.
Your task is to answer user queries about road safety interventions.

You will be given a context of retrieved database information and a user query.
Your response MUST follow these rules:

<context>
{context_str}
</context>

<query>
{query_str}
</query>

---
INSTRUCTIONS:

1.  **CRITICAL RELEVANCE CHECK:**
    * First, you *must* determine if the provided <context> *directly and meaningfully* answers the user's <query>.
    * Ask yourself: "Does this context block contain the *specific intervention or specification* for the problem in the query?"
    * A simple *mention* of a keyword is NOT a relevant answer.

2.  **If the context IS RELEVANT:**
    * You MUST generate a response in this strict format:
    * 1.  **Direct Answer:** Provide a detailed, comprehensive answer to the user's query by **summarizing the key information, specifications, dimensions, and procedures** found *directly* in the context. Explain the "what," "why," and "how" from the text.
    * 2.  **Source Code:** Extract the exact 'code' (e.g., IRC:67-2022) from the context that supports your answer.
    * 3.  **Clause:** Extract the exact 'clause' (e.g., 14.4) from the context that supports your answer.

3.  **If the context IS NOT RELEVANT:**
    * You MUST discard the context.
    * You MUST generate *only* this exact sentence:
    * "The provided database does not contain information on this specific topic."

4.  **NEVER** use any outside knowledge or make assumptions.
"""


def create_query_engine(llm_to_use):
    """
    Builds and returns the complete RAG query engine.
    """
    print("--- Initializing Query Engine ---")

    # CONFIGURE GLOBAL SETTINGS
    print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = llm_to_use
    print(f"LLM set to: {llm_to_use.metadata.model_name}")

    #  LOAD THE VECTOR DATABASE
    print(f"Loading vector database from: {DB_PERSIST_DIR}")
    db = chromadb.PersistentClient(path=DB_PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )
    print("Successfully loaded index from vector database.")


    # QUERY ENGINE 
    query_engine = index.as_query_engine(

        similarity_top_k=5,  
 
        text_qa_template=PromptTemplate(PROMPT_TEMPLATE)
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
            sys.exit(1) 


if __name__ == "__main__":
    print("Testing LOCAL (Ollama) query engine...")
    local_llm = get_llm()
    query_engine = create_query_engine(local_llm)
    
    print("\n--- Test Query ---")
    test_query = "What are the rules for a STOP sign?"
    print(f"Query: '{test_query}'")
    
    response = query_engine.query(test_query)
    
    print("\n--- Test Response ---")
    print(response)
    print("\n--- Source Nodes ---")
    for node in response.source_nodes:
        print(f"Score: {node.score:.4f}")
        print(f"Metadata: {node.metadata}")
        print(f"Text: {node.node.get_content()[:100]}...")
        print("---")