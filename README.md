#  AeroDoc-AI: Aerospace Technical Document Intelligence

This project is a specialized **Retrieval-Augmented Generation (RAG) assistant** designed for aerospace engineering documentation, turbopump systems, and turbine engine manuals.

The idea behind this was simple — instead of manually digging through dense technical manuals, I wanted to build an **intelligent, grounded assistant that instantly pulls exact technical answers with precise page-level citations**.

So I built this application that processes technical PDFs, vectorizes their content, and walks through the entire RAG pipeline:
from raw documents → vector retrieval → Groq Llama 3 inference → real-time evaluation metrics.

---

##  Key Features

* **Grounded Aerospace Intelligence**: Answers technical queries using context retrieved strictly from uploaded engineering manuals and technical papers.
* **Instant Citation Tracking**: Displays explicit page-level and document-level citations (`[Filename, Page X]`) for every answer.
* **Interactive Evaluation Metrics**: Built-in real-time calculation for **Retrieval Relevance** and **Answer Faithfulness** without extra API overhead.
* **Groq Llama 3 Inference**: Native integration using `llama-3.3-70b-versatile` with automatic fallback to `llama3-8b-8192`.
* **Zero Cost Architecture**: Uses local embeddings (`sentence-transformers`), FAISS CPU vector indices, and free-tier Groq API access.
* **WhatsApp-Style Dark Mode UI**: Clean Streamlit frontend customized with dark theme chat bubbles and sidebar key management.

---

##  Tech Stack

* Python
* Streamlit
* FAISS-CPU
* Sentence-Transformers
* Groq API (Llama 3 Models)
* PyPDF2 / LangChain Text Splitters

---

##  How to run this locally

### 1. Clone the Repository
```bash
git clone [https://github.com/Venukomkala/AeroDoc-AI.git](https://github.com/Venukomkala/AeroDoc-AI.git)
cd AeroDoc-AI

python -m venv venv
venv\Scripts\activate

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python tests/test_vector_store.py

streamlit run app/streamlit_app.py
