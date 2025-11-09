# National Road Safety Intervention GPT

This project is a submission for the National Road Safety Hackathon 2025. It's an AI-powered tool that answers road safety questions based on a provided database, using a high-performance Retrieval-Augmented Generation (RAG) system.

**Theme:** AI in Road Safety
**Problem Statement:** Road Safety Intervention GPT

---

## Live Demo

You can access the live, zero-setup demo here:
**[https://roadsafetygptnathack.streamlit.app/]**

*(Note: The live demo runs on Streamlit's free cloud tier and uses the Gemini 1.5 Pro API.)*

---

## Tech Stack

* **Framework:** LlamaIndex & Streamlit
* **Vector Database:** ChromaDB
* **Embedding Model:** `BAAI/bge-large-en-v1.5`
* **Reranker Model:** `BAAI/bge-reranker-v2-m3`
* **LLMs (Hybrid):**
    * **Local:** `Meta Llama 3 8B` (via Ollama)
    * **Cloud:** `Google Gemini 1.5 Pro`

---

## How to Run Locally

This project can be run 100% locally, for free, using the open-source Llama 3 model.

### Prerequisites

1.  **Python 3.10+**
2.  **Ollama:** Download and install from [https://ollama.com](https://ollama.com)
3.  **Git:** Download and install from [https://git-scm.com/](https://git-scm.com/)

### Step 1: Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/Road_Safety_GPT.git](https://github.com/YOUR_USERNAME/Road_Safety_GPT.git)
cd Road_Safety_GPT
