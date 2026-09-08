import os
import sys

# Ensure project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retriever import AeroRetriever
from src.generator import AeroGenerator

def run_test():
    print("=== AeroDoc-AI: End-to-End Generation Test ===")

    # 1. Retrieve Context
    retriever = AeroRetriever(processed_data_dir="data/processed")
    query = "What are the main causes of turbine blade thermal stresses in rocket engines?"
    
    retrieved_chunks = retriever.retrieve(query, top_k=3)
    formatted_context = retriever.format_context(retrieved_chunks)

    # 2. Generate Grounded Answer
    generator = AeroGenerator(model_name="gpt-4o-mini")
    print("\n[Generator] Sending prompt to LLM...")
    answer = generator.generate_answer(query, formatted_context)

    print("\n=== USER QUERY ===")
    print(query)

    print("\n=== GENERATED RESPONSE ===")
    print(answer)

if __name__ == "__main__":
    run_test()