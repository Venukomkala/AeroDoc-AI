import os
import sys

# Ensure project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retriever import AeroRetriever

def run_test():
    print("=== AeroDoc-AI: Testing Semantic Retrieval ===")

    # Initialize retriever
    retriever = AeroRetriever(processed_data_dir="data/processed")

    # Sample technical query
    query = "What are the main causes of turbine blade thermal stresses in rocket engines?"
    
    # Perform retrieval
    results = retriever.retrieve(query, top_k=3)

    print(f"\nQUERY: '{query}'\n")
    print("=== RETRIEVED TOP CHUNKS ===")
    
    for item in results:
        print(f"\n[Rank {item['rank']}] Score: {item['score']}")
        print(f"File : {item['filename']}")
        print(f"Page : {item['page_number']}")
        print(f"Text : {item['text'][:250]}...")

    print("\n=== FORMATTED CONTEXT BLOCK ===")
    print(retriever.format_context(results)[:600] + "...")

if __name__ == "__main__":
    run_test()