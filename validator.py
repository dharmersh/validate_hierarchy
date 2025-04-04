import json
import numpy as np
from typing import List, Dict
from utils.embedding_utils import EmbeddingGenerator
from utils.similarity_utils import SimilarityCalculator

class ParentChildValidator:
    def __init__(self, data_path: str, embeddings_path: str):
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        self.embedding_generator = EmbeddingGenerator()
        self.data = self._load_data()
        self.embeddings = self._load_or_generate_embeddings()
    
    def _load_data(self) -> List[Dict]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Input data must be a JSON array")
            return data
    
    def _load_or_generate_embeddings(self) -> Dict[str, np.ndarray]:
        saved_embeddings = self.embedding_generator.load_embeddings(self.embeddings_path)
        if saved_embeddings is not None:
            return saved_embeddings
        
        root_descriptions = [item.get('root_description', '') for item in self.data]
        parent_summaries = [item.get('parent_short_summary', '') for item in self.data]
        
        embeddings = {
            'root': self.embedding_generator.generate_embeddings(root_descriptions),
            'parent': self.embedding_generator.generate_embeddings(parent_summaries)
        }
        
        self.embedding_generator.save_embeddings(embeddings, self.embeddings_path)
        return embeddings
    
    def validate_relationships(self, similarity_threshold: float = 0.6) -> List[Dict]:
        results = []
        
        for idx, item in enumerate(self.data):
            if not item.get('parent_name'):
                continue
            
            root_embedding = self.embeddings['root'][idx]
            parent_embedding = self.embeddings['parent'][idx]
            
            similarity_score = SimilarityCalculator.calculate_similarity(root_embedding, parent_embedding)
            
            all_parent_indices = [i for i, x in enumerate(self.data) if x.get('parent_name') and i != idx]
            parent_candidates = [self.data[i] for i in all_parent_indices]
            parent_candidate_embeddings = [self.embeddings['parent'][i] for i in all_parent_indices]
            
            all_matches = []
            for match_idx, (embedding, data) in enumerate(zip(parent_candidate_embeddings, parent_candidates)):
                if embedding is not None:
                    sim = SimilarityCalculator.calculate_similarity(root_embedding, embedding)
                    all_matches.append((sim, match_idx, data))
            
            all_matches.sort(reverse=True, key=lambda x: x[0])
            
            suggestions = []
            for sim, match_idx, _ in all_matches:
                parent_data = parent_candidates[match_idx]
                suggestions.append({
                    'parent_key': parent_data.get('parnet_key'),
                    'parent_name': parent_data.get('parent_name'),
                    'similarity_score': float(sim),
                    'improvement': float(sim) - similarity_score
                })
            
            results.append({
                'root_key': item.get('root_key'),
                'root_name': item.get('root_name'),
                'current_score': float(similarity_score),
                'current_parent': {
                    'parent_key': item.get('parnet_key'),
                    'parent_name': item.get('parent_name'),
                    'similarity_score': float(similarity_score)
                },
                'all_suggestions': suggestions,
                'validation': 'VALID' if similarity_score >= similarity_threshold else 'INVALID',
                'validation_status': 'PASS' if similarity_score >= similarity_threshold else 'FAIL'
            })
        
        return results