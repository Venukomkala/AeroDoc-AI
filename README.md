#  AeroDoc-AI: Aerospace Technical Document Intelligence

A high-performance, grounded Retrieval-Augmented Generation (RAG) assistant designed for aerospace engineering documentation, turbopump systems, and turbine engine manuals. Built with a sleek WhatsApp-inspired interface, vector search via FAISS, and inference powered by Groq's fast Llama 3 models.

---

##  Key Features

* **Grounded Aerospace Intelligence**: Answers technical queries using context retrieved strictly from uploaded engineering manuals and technical papers.
* **Instant Citation Tracking**: Displays explicit page-level and document-level citations (`[Filename, Page X]`) for every answer.
* **Interactive Evaluation Metrics**: Built-in real-time calculation for **Retrieval Relevance** and **Answer Faithfulness** without extra API overhead.
* **Groq Llama 3 Inference**: Native integration using `llama-3.3-70b-versatile` with automatic fallback to `llama3-8b-8192`.
* **Zero Cost Architecture**: Uses local embeddings (`sentence-transformers`), FAISS CPU vector indices, and free-tier Groq API access.
* **WhatsApp-Style Dark Mode UI**: Clean Streamlit frontend customized with dark theme chat bubbles and sidebar key management.

---

1. Clone the Repository

git clone [https://github.com/Venukomkala/AeroDoc-AI.git](https://github.com/Venukomkala/AeroDoc-AI.git)
cd AeroDoc-AI


2. Create and Activate Virtual Environment

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

3. Install Requirements

pip install -r requirements.txt

Running the Application

python tests/test_vector_store.py

Launch the Streamlit Web App

streamlit run app/streamlit_app.py


Step 3: Configure Your API Key
Get a free Groq API key from console.groq.com/keys.