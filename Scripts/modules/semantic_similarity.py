from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from typing import List, Tuple, Optional

class SemanticSimilarityScorer:
    """
    A module that computes semantic similarity scores between LLM responses 
    and retrieved RAG context to detect potential hallucinations.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic similarity model.
        
        Args:
            model_name: SentenceTransformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"✅ Semantic Similarity Model loaded: {model_name}")
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score between 0 and 1
        """
        # Encode texts
        embeddings1 = self.model.encode(text1, convert_to_tensor=True)
        embeddings2 = self.model.encode(text2, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity = util.cos_sim(embeddings1, embeddings2)
        
        return similarity.item()
    
    def assess_response(
        self, 
        llm_response: str, 
        rag_contexts: List[str],
        method: str = "max_similarity"
    ) -> dict:
        """
        Assess an LLM response against multiple RAG context chunks.
        
        Args:
            llm_response: The LLM-generated response
            rag_contexts: List of context chunks retrieved from RAG
            method: Scoring method ('max_similarity', 'average', 'weighted_average')
            
        Returns:
            Dictionary containing:
                - overall_score: Semantic similarity score (0-1)
                - chunk_scores: Similarity scores for each context chunk
                - max_similar_chunk: The most similar context chunk
                - is_hallucination: Boolean flag (based on threshold)
                - assessment: Textual assessment
        """
        if not rag_contexts:
            return {
                "overall_score": 0.0,
                "chunk_scores": [],
                "max_similar_chunk": "",
                "is_hallucination": True,
                "assessment": "No RAG context provided for comparison."
            }
        
        # Compute similarity with each context chunk
        chunk_scores = []
        for context in rag_contexts:
            score = self.compute_similarity(llm_response, context)
            chunk_scores.append({
                "context": context[:200] + "..." if len(context) > 200 else context,
                "score": score
            })
        
        # Calculate overall score based on method
        scores = [item["score"] for item in chunk_scores]
        
        if method == "max_similarity":
            overall_score = max(scores)
        elif method == "average":
            overall_score = np.mean(scores)
        elif method == "weighted_average":
            # Weight by context length (longer context gets more weight)
            weights = [len(ctx["context"]) for ctx in chunk_scores]
            weights = [w/sum(weights) for w in weights]  # Normalize
            overall_score = np.average(scores, weights=weights)
        else:
            overall_score = max(scores)  # Default to max
        
        # Find the most similar chunk
        max_idx = scores.index(max(scores))
        max_similar_chunk = chunk_scores[max_idx]["context"]
        
        # Determine if response is hallucinated
        # Lower similarity = more likely hallucination
        is_hallucination = overall_score < 0.4  # Adjustable threshold
        
        # Generate assessment text
        if overall_score >= 0.7:
            assessment = "Response is highly consistent with the provided context."
        elif overall_score >= 0.5:
            assessment = "Response shows moderate consistency with context."
        elif overall_score >= 0.3:
            assessment = "Response shows weak consistency. Potential partial hallucination."
        else:
            assessment = "Response appears to be hallucinated or unrelated to context."
        
        return {
            "overall_score": float(overall_score),
            "chunk_scores": chunk_scores,
            "max_similar_chunk": max_similar_chunk,
            "is_hallucination": is_hallucination,
            "assessment": assessment
        }
    
    def batch_assess(
        self, 
        llm_responses: List[str], 
        rag_contexts_list: List[List[str]]
    ) -> List[dict]:
        """
        Assess multiple LLM responses in batch.
        
        Args:
            llm_responses: List of LLM responses
            rag_contexts_list: List of context lists for each response
            
        Returns:
            List of assessment results
        """
        results = []
        for response, contexts in zip(llm_responses, rag_contexts_list):
            result = self.assess_response(response, contexts)
            results.append(result)
        return results
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract individual claims/statements from a response for fine-grained analysis.
        (Simple implementation - can be enhanced with more sophisticated NLP)
        Args:
            text: LLM response text
        Returns:
            List of individual claims
        """
        # Simple sentence splitting for demonstration
        # In production, use NER or claim extraction models
        import re
        sentences = re.split(r'[.!?]+', text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims
    
    def assess_individual_claims(
        self, 
        llm_response: str, 
        rag_contexts: List[str]
    ) -> dict:
        """
        Assess individual claims within a response for more granular hallucination detection.
        
        Args:
            llm_response: The LLM-generated response
            rag_contexts: List of context chunks
            
        Returns:
            Dictionary with claim-level analysis
        """
        claims = self.extract_claims(llm_response)
        
        claim_assessments = []
        for claim in claims:
            claim_result = self.assess_response(claim, rag_contexts)
            claim_assessments.append({
                "claim": claim,
                "score": claim_result["overall_score"],
                "is_hallucination": claim_result["is_hallucination"]
            })
        
        # Calculate overall metrics
        claim_scores = [c["score"] for c in claim_assessments]
        hallucination_count = sum(1 for c in claim_assessments if c["is_hallucination"])
        
        return {
            "total_claims": len(claims),
            "hallucinated_claims": hallucination_count,
            "avg_claim_score": np.mean(claim_scores) if claim_scores else 0,
            "claim_assessments": claim_assessments
        }


# Convenience function for integration with existing code
def semantic_similarity_run(llm_response: str, rag_contexts: List[str]) -> dict:
    """
    Runs the semantic similarity module.
    
    Args:
        llm_response: LLM-generated response
        rag_contexts: List of RAG context chunks
        
    Returns:
        Assessment results dictionary
    """
    scorer = SemanticSimilarityScorer()
    return scorer.assess_response(llm_response, rag_contexts)