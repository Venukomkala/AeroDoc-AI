import os
import sys

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import RAGEvaluator

def run_test():
    print("=== AeroDoc-AI: Testing Evaluation Metrics ===")

    evaluator = RAGEvaluator()

    query = "What causes thermal stress in turbine blades?"
    
    mock_chunks = [
        {"text": "Thermal stress in turbine blades is caused by steep temperature gradients across internal cooling channels during engine start."},
        {"text": "High temperature gas flows over nozzle guide vanes causing mechanical wear."}
    ]
    
    context = "\n".join([c["text"] for c in mock_chunks])
    
    response = "Thermal stress in turbine blades is caused by steep temperature gradients across cooling channels."

    results = evaluator.run_full_evaluation(query, response, mock_chunks, context)

    print("\n--- EVALUATION METRICS ---")
    print(f"Retrieval Relevance : {results['retrieval_relevance'] * 100:.1f}%")
    print(f"Answer Faithfulness : {results['faithfulness'] * 100:.1f}%")
    print("--------------------------")

if __name__ == "__main__":
    run_test()