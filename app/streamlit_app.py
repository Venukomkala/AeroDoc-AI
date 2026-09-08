import os
import sys
import streamlit as st

# Ensure project root is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retriever import AeroRetriever
from src.generator import AeroGenerator
from src.evaluation import RAGEvaluator

# Page Configuration
st.set_page_config(
    page_title="AeroDoc-AI Assistant",
    page_icon="✈️",
    layout="wide"
)

# WhatsApp-style Custom CSS
st.markdown("""
<style>
    /* Dark Theme Background */
    .stApp {
        background-color: #0b141a;
        color: #e9edef;
    }
    
    /* User Message Bubble */
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #005c4b !important;
        border-radius: 12px 12px 0px 12px !important;
        color: #ffffff !important;
        margin-left: auto !important;
        max-width: 80% !important;
        padding: 10px 14px !important;
    }
    
    /* Assistant Message Bubble */
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #202c33 !important;
        border-radius: 12px 12px 12px 0px !important;
        color: #e9edef !important;
        margin-right: auto !important;
        max-width: 80% !important;
        padding: 10px 14px !important;
    }

    /* Input area styling */
    div[data-testid="stChatInput"] input {
        background-color: #2a3942 !important;
        color: #e9edef !important;
        border-radius: 20px !important;
        border: none !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111b21 !important;
        border-right: 1px solid #222d34;
    }
</style>
""", unsafe_allow_html=True)

# Title & Description
st.title("✈️ AeroDoc-AI: Technical Document Intelligence")
st.caption("Ask complex aerospace engineering queries with grounded document retrieval and citation tracking.")

# Sidebar Configuration
st.sidebar.header("⚙️ System Configuration")

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

with st.sidebar.form("api_key_form"):
    input_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.groq_api_key,
        help="Enter your free Groq API key starting with 'gsk_' to enable Llama 3.1 inference."
    )
    submit_key = st.form_submit_button("Save API Key")
    if submit_key:
        st.session_state.groq_api_key = input_key.strip()
        st.success("Key saved successfully!")

groq_api_key = st.session_state.groq_api_key

top_k = st.sidebar.slider("Top-K Chunks to Retrieve", min_value=1, max_value=5, value=3)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load Pipelines
@st.cache_resource
def load_retriever():
    return AeroRetriever(processed_data_dir="data/processed")

try:
    retriever = load_retriever()
    evaluator = RAGEvaluator()
    st.sidebar.success("FAISS Index loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Failed to load vector store: {e}")
    retriever = None

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Referenced Sources & Citations"):
                for source in message["sources"]:
                    st.write(f"**[{source['filename']} | Page {source['page_number']}]** (Score: {source['score']})")
                    st.caption(source["text"])

# Chat Input & Response Loop
if prompt := st.chat_input("Ask a question about your aerospace manuals or technical papers..."):
    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not retriever:
        with st.chat_message("assistant"):
            st.error("Retriever is not initialized. Run `python tests/test_vector_store.py` in your terminal first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching technical context & generating response..."):
                # 1. Retrieve Chunks
                retrieved_chunks = retriever.retrieve(prompt, top_k=top_k)
                formatted_context = retriever.format_context(retrieved_chunks)

                # 2. Generate Answer
                if groq_api_key.strip():
                    generator = AeroGenerator(api_key=groq_api_key)
                    answer = generator.generate_answer(prompt, formatted_context)
                else:
                    answer = (
                        "⚠️ **Groq API Key Missing**: Please enter your free key (`gsk_...`) in the sidebar to generate AI responses.\n\n"
                        "**Retrieved Document Excerpts:**\n" + formatted_context
                    )

                # 3. Evaluate Metrics
                eval_metrics = evaluator.run_full_evaluation(prompt, answer, retrieved_chunks, formatted_context)

                # Render Answer
                st.markdown(answer)

                # Render Evaluation Metrics
                st.write("---")
                col1, col2 = st.columns(2)
                col1.metric("Retrieval Relevance", f"{eval_metrics['retrieval_relevance'] * 100:.1f}%")
                col2.metric("Answer Faithfulness", f"{eval_metrics['faithfulness'] * 100:.1f}%")

                # Render Source Citations
                with st.expander("📚 Referenced Sources & Citations"):
                    for chunk in retrieved_chunks:
                        st.write(f"**[{chunk['filename']} | Page {chunk['page_number']}]** (Score: {chunk['score']})")
                        st.caption(chunk["text"])

                # Store assistant response with metadata in history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved_chunks
                })