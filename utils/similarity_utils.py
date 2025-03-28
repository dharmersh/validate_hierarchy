from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import SIMILARITY_THRESHOLD, TOP_N_SUGGESTIONS
from typing import List, Dict, Optional
# class SimilarityCalculator:
#     def __init__(self):
#         pass

#     def calculate_similarity(self, embedding1, embedding2):
#         """Calculate cosine similarity between two embeddings"""
#         if embedding1 is None or embedding2 is None:
#             return 0
        
#         # Convert tensors to numpy arrays if needed
#         if hasattr(embedding1, 'numpy'):
#             embedding1 = embedding1.numpy()
#         if hasattr(embedding2, 'numpy'):
#             embedding2 = embedding2.numpy()
            
#         # Ensure embeddings are 2D arrays
#         if len(embedding1.shape) == 1:
#             embedding1 = embedding1.reshape(1, -1)
#         if len(embedding2.shape) == 1:
#             embedding2 = embedding2.reshape(1, -1)
            
#         return cosine_similarity(embedding1, embedding2)[0][0]

#     def find_best_matches(self, target_embedding, candidate_embeddings, candidate_data):
#         """Find best matching parents based on similarity"""
#         if target_embedding is None:
#             return []
            
#         similarities = []
#         for idx, (embedding, data) in enumerate(zip(candidate_embeddings, candidate_data)):
#             if embedding is not None:
#                 try:
#                     sim = self.calculate_similarity(target_embedding, embedding)
#                     similarities.append((sim, idx))
#                 except Exception as e:
#                     print(f"Similarity calculation failed for index {idx}: {str(e)}")
#                     continue
        
#         # Sort by similarity score descending
#         similarities.sort(reverse=True, key=lambda x: x[0])
        
#         # Filter by threshold and return top N
#         filtered = [x for x in similarities if x[0] >= SIMILARITY_THRESHOLD]
#         return filtered[:TOP_N_SUGGESTIONS]

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

class SimilarityCalculator:
    @staticmethod
    def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        return cosine_similarity([embedding1], [embedding2])[0][0]
    
    @staticmethod
    def find_top_matches(
        target_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        candidate_data: List[Dict],
        top_n: int = 3,
        threshold: float = 0.6
    ) -> List[Tuple[float, int, Dict]]:
        """
        Find top matching candidates based on similarity score
        
        Returns:
            List of tuples (similarity_score, index, candidate_data)
        """
        similarities = []
        for idx, (embedding, data) in enumerate(zip(candidate_embeddings, candidate_data)):
            if embedding is not None:
                sim = SimilarityCalculator.calculate_similarity(target_embedding, embedding)
                if sim >= threshold:
                    similarities.append((sim, idx, data))
        
        # Sort by similarity score (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])
        return similarities[:top_n]