# from sentence_transformers import SentenceTransformer
# import pickle
# import os
# from config import EMBEDDING_MODEL


import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

from typing import List, Dict, Optional

#class EmbeddingGenerator:
#     def __init__(self):
#         self.model = SentenceTransformer(EMBEDDING_MODEL)
    
#     def generate_embeddings(self, texts):
#         return self.model.encode(texts, convert_to_tensor=True)
    
#     def save_embeddings(self, embeddings, file_path):
#         # Create directory if it doesn't exist
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, 'wb') as f:
#             pickle.dump(embeddings, f)
    
#     def load_embeddings(self, file_path):
#         if os.path.exists(file_path):
#             with open(file_path, 'rb') as f:
#                 return pickle.load(f)
#         return None

    
class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of text inputs"""
        return self.model.encode(texts, convert_to_tensor=False)
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], file_path: str):
        """Save embeddings dictionary to file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
    
    def load_embeddings(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Load embeddings from file if exists"""
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return None