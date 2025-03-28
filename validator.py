import json
import os
from typing import List, Dict
from utils.embedding_utils import EmbeddingGenerator
from utils.similarity_utils import SimilarityCalculator
from config import SIMILARITY_THRESHOLD
import pandas as pd
from xlsxwriter import Workbook

class ParentChildValidator:
    def __init__(self, data_path: str, embeddings_path: str):
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        self.embedding_generator = EmbeddingGenerator()
        self.similarity_calculator = SimilarityCalculator()
        self.data = self._load_data()
        self.embeddings = self._load_or_generate_embeddings()
    
    def _load_data(self) -> List[Dict]:
        """Load and validate the input JSON data"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Input data must be a JSON array")
                return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in input file")
    
    def _load_or_generate_embeddings(self) -> Dict[str, List]:
        """Load precomputed embeddings or generate new ones if needed"""
        # Try to load saved embeddings
        saved_embeddings = self.embedding_generator.load_embeddings(self.embeddings_path)
        
        if saved_embeddings is not None:
            return saved_embeddings
        
        # Generate new embeddings if not found
        print("Generating new embeddings...")
        root_descriptions = [item.get('root_description', '') or '' for item in self.data]
        parent_summaries = [item.get('parent_short_summary', '') or '' for item in self.data]
        
        root_embeddings = self.embedding_generator.generate_embeddings(root_descriptions)
        parent_embeddings = self.embedding_generator.generate_embeddings(parent_summaries)
        
        embeddings = {
            'root': root_embeddings,
            'parent': parent_embeddings
        }
        
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        self.embedding_generator.save_embeddings(embeddings, self.embeddings_path)
        return embeddings
    
    def validate_relationships(self) -> List[Dict]:
        """Validate all parent-child relationships in the data"""
        results = []
        
        for idx, item in enumerate(self.data):
            if not item.get('parent_name'):
                continue  # Skip items with no parent
                
            root_embedding = self.embeddings['root'][idx]
            parent_embedding = self.embeddings['parent'][idx]
            
            # Calculate similarity between current parent and root
            similarity_score = self.similarity_calculator.calculate_similarity(
                root_embedding, parent_embedding
            )
            
            # Find alternative parent suggestions
            all_parent_indices = [
                i for i, x in enumerate(self.data) 
                if x.get('parent_name') and i != idx
            ]
            parent_candidates = [self.data[i] for i in all_parent_indices]
            parent_candidate_embeddings = [self.embeddings['parent'][i] for i in all_parent_indices]
            
            best_matches = self.similarity_calculator.find_best_matches(
                root_embedding,
                parent_candidate_embeddings,
                parent_candidates
            )
            
            suggestions = []
            for score, match_idx in best_matches:
                parent_data = parent_candidates[match_idx]
                suggestions.append({
                    'parent_key': parent_data.get('parnet_key'),
                    'parent_name': parent_data.get('parent_name'),
                    'similarity_score': float(score)
                })
            
            results.append({
                'root_key': item.get('root_key'),
                'root_name': item.get('root_name'),
                'root_description': item.get('root_description'),
                'current_parent': {
                    'parent_key': item.get('parnet_key'),
                    'parent_name': item.get('parent_name'),
                    'similarity_score': float(similarity_score)
                },
                'suggested_parents': suggestions,
                'validation': 'VALID' if similarity_score >= SIMILARITY_THRESHOLD else 'INVALID',
                'validation_status': 'PASS' if similarity_score >= SIMILARITY_THRESHOLD else 'FAIL'
            })
        
        return results
    
    def generate_report(self, validation_results: List[Dict]) -> str:
        """Generate a human-readable validation report"""
        report_lines = []
        
        for result in validation_results:
            report_lines.append(f"\n{'=' * 80}")
            report_lines.append(f"Root Key: {result['root_key']}")
            report_lines.append(f"Root Name: {result['root_name']}")
            report_lines.append(f"Root Description: {result['root_description']}")
            
            report_lines.append("\nCurrent Parent:")
            report_lines.append(f"  - Name: {result['current_parent']['parent_name']}")
            report_lines.append(f"  - Similarity Score: {result['current_parent']['similarity_score']:.2f}")
            report_lines.append(f"  - Validation: {result['validation']} ({result['validation_status']})")
            
            if result['suggested_parents']:
                report_lines.append("\nSuggested Alternative Parents:")
                for suggestion in result['suggested_parents']:
                    report_lines.append(
                        f"  - {suggestion['parent_name']} "
                        f"(Score: {suggestion['similarity_score']:.2f})"
                    )
            else:
                report_lines.append("\nNo suitable alternative parents found.")
        
        report_lines.append(f"\n{'=' * 80}")
        report_lines.append("\nValidation Summary:")
        total = len(validation_results)
        passed = sum(1 for r in validation_results if r['validation_status'] == 'PASS')
        report_lines.append(f"Total relationships validated: {total}")
        report_lines.append(f"Passed: {passed} ({(passed/total)*100:.1f}%)")
        report_lines.append(f"Failed: {total - passed} ({(1-passed/total)*100:.1f}%)")
        
        return "\n".join(report_lines)

    def generate_excel_report(self, validation_results: List[Dict]) -> pd.DataFrame:
        """Prepare data for Excel report"""
        report_data = []
        
        for result in validation_results:
            suggestions = "; ".join(
                f"{s['parent_name']} (Score: {s['similarity_score']:.2f})"
                for s in result['suggested_parents']
            ) if result['suggested_parents'] else "No suitable alternatives found"
            
            report_data.append({
                'Root Key': result['root_key'],
                'Root Name': result['root_name'],
                'Root Description': result['root_description'],
                'Current Parent': result['current_parent']['parent_name'],
                'Parent Key': result['current_parent']['parent_key'],
                'Similarity Score': result['current_parent']['similarity_score'],
                'Validation': result['validation'],
                'Status': result['validation_status'],
                'Suggested Parents': suggestions
            })
        
        return pd.DataFrame(report_data)
    
def save_to_excel(df, summary_df, report_path):
    """Save data to Excel with fallback engine support"""
    try:
        # Try with xlsxwriter first
        with pd.ExcelWriter(report_path, engine='xlsxwriter') as writer:
            _write_excel(writer, df, summary_df)
    except ImportError:
        try:
            # Fallback to openpyxl
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                _write_excel(writer, df, summary_df)
        except Exception as e:
            raise Exception(f"Failed to create Excel file: {str(e)}")
        
def _write_excel(writer, df, summary_df):
    """Helper function to write Excel data"""
    df.to_excel(writer, sheet_name='Validation Results', index=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Basic formatting available in both engines
    if writer.engine == 'xlsxwriter':
        workbook = writer.book
        worksheet = writer.sheets['Validation Results']
        
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, max_len + 2)
    
    elif writer.engine == 'openpyxl':
        worksheet = writer.sheets['Validation Results']
        for column_cells in worksheet.columns:
            length = max(len(str(cell.value)) for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = length + 2

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'input.json')
    EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'embeddings', 'embeddings.pkl')
    REPORT_PATH = os.path.join(BASE_DIR, 'validation_report.xlsx')
    
    print("Starting validation process...")
    try:
        # Ensure directories exist
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        
        validator = ParentChildValidator(
            data_path=DATA_PATH,
            embeddings_path=EMBEDDINGS_PATH
        )
        results = validator.validate_relationships()
        df = validator.generate_excel_report(results)
        
        # Calculate summary statistics
        total = len(df)
        passed = len(df[df['Status'] == 'PASS'])
        summary_data = {
            'Total Relationships': [total],
            'Passed': [passed],
            'Failed': [total - passed],
            'Pass Rate': [f"{(passed/total)*100:.1f}%"]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Save to Excel
        with pd.ExcelWriter(REPORT_PATH, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Validation Results', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Validation Results']
            summary_sheet = writer.sheets['Summary']
            
            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#4472C4',
                'font_color': 'white',
                'border': 1
            })
            
            # Apply header format
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            for col_num, value in enumerate(summary_df.columns.values):
                summary_sheet.write(0, col_num, value, header_format)
            
            # Auto-adjust column widths
            for i, col in enumerate(df.columns):
                max_len = max(
                    df[col].astype(str).map(len).max(),
                    len(col))
                worksheet.set_column(i, i, max_len + 2)
            
            for i, col in enumerate(summary_df.columns):
                max_len = max(
                    summary_df[col].astype(str).map(len).max(),
                    len(col))
                summary_sheet.set_column(i, i, max_len + 2)
            
            # Add conditional formatting
            worksheet.conditional_format(
                'G2:G1000',  # Validation column
                {
                    'type': 'cell',
                    'criteria': 'equal to',
                    'value': '"VALID"',
                    'format': workbook.add_format({'bg_color': '#C6EFCE'})
                })
            
            worksheet.conditional_format(
                'G2:G1000',
                {
                    'type': 'cell',
                    'criteria': 'equal to',
                    'value': '"INVALID"',
                    'format': workbook.add_format({'bg_color': '#FFC7CE'})
                })
        
        print(f"\nValidation completed successfully!")
        print(f"Excel report saved to: {REPORT_PATH}")
        
    except Exception as e:
        print(f"\nError during validation: {str(e)}")
        raise