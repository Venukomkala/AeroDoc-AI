import re

class RAGEvaluator:
    def __init__(self):
        pass

    def evaluate_retrieval_relevance(self, query, retrieved_chunks):
        """
        Calculates word overlap precision between query terms and retrieved chunks.
        Returns a relevance score normalized between 0.0 and 1.0.
        """
        if not retrieved_chunks:
            return 0.0

        query_tokens = set(re.findall(r'\w+', query.lower()))
        # Remove common short stopwords for better accuracy
        stopwords = {"what", "is", "are", "the", "in", "of", "and", "a", "to", "for", "on", "with"}
        query_tokens = query_tokens - stopwords

        if not query_tokens:
            return 0.0

        scores = []
        for chunk in retrieved_chunks:
            chunk_text = chunk.get("text", "").lower()
            chunk_tokens = set(re.findall(r'\w+', chunk_text))
            
            overlap = query_tokens.intersection(chunk_tokens)
            score = len(overlap) / len(query_tokens)
            scores.append(score)

        return round(sum(scores) / len(scores), 4)

    def evaluate_faithfulness(self, response_text, context_text):
        """
        Evaluates the proportion of sentence claims in the generated response 
        that overlap with key vocabulary in the retrieved context.
        """
        if not response_text or "LLM generation failed" in response_text:
            return 0.0

        sentences = [s.strip() for s in re.split(r'[.!?]', response_text) if len(s.strip()) > 10]
        if not sentences:
            return 1.0

        context_tokens = set(re.findall(r'\w+', context_text.lower()))
        stopwords = {"this", "that", "there", "is", "are", "was", "were", "be", "been", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        grounded_count = 0
        for sentence in sentences:
            sent_tokens = set(re.findall(r'\w+', sentence.lower())) - stopwords
            if not sent_tokens:
                continue
            
            overlap = sent_tokens.intersection(context_tokens)
            # If at least 40% of non-stopword tokens exist in context, treat sentence as grounded
            if (len(overlap) / len(sent_tokens)) >= 0.4:
                grounded_count += 1

        return round(grounded_count / len(sentences), 4)

    def run_full_evaluation(self, query, response_text, retrieved_chunks, context_text):
        """
        Runs both evaluations and returns a combined summary dictionary.
        """
        retrieval_score = self.evaluate_retrieval_relevance(query, retrieved_chunks)
        faithfulness_score = self.evaluate_faithfulness(response_text, context_text)

        return {
            "query": query,
            "retrieval_relevance": retrieval_score,
            "faithfulness": faithfulness_score,
            "chunks_retrieved": len(retrieved_chunks)
        }