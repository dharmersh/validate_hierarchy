from sentence_transformers import SentenceTransformer
import pickle
import os
from config import EMBEDDING_MODEL

class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
    
    def generate_embeddings(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)
    
    def save_embeddings(self, embeddings, file_path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
    
    def load_embeddings(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return None