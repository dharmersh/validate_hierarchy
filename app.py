import streamlit as st
import pandas as pd
from validator import ParentChildValidator
from io import BytesIO
from datetime import datetime

# Workaround for PyTorch issue
if 'streamlit' in globals():
    import torch
    torch.classes.__path__ = []

def prepare_data(results):
    current = []
    suggestions = []
    
    for result in results:
        current.append({
            'Root Key': result['root_key'],
            'Root Name': result['root_name'],
            'Current Parent': result['current_parent']['parent_name'],
            'Score': result['current_parent']['similarity_score'],
            'Status': result['validation']
        })
        
        # Use the correct key based on validator's output structure
        suggestions_key = 'suggested_parents' if 'suggested_parents' in result else 'all_suggestions'
        for suggestion in result.get(suggestions_key, []):
            suggestions.append({
                'Root Key': result['root_key'],
                'Root Name': result['root_name'],
                'Suggested Parent': suggestion['parent_name'],
                'Similarity Score': suggestion['similarity_score']
            })
    
    return pd.DataFrame(current), pd.DataFrame(suggestions)

def to_excel(current_df, suggestions_df):
    """Convert both dataframes to an Excel file with multiple sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        current_df.to_excel(writer, sheet_name='Current Relationships', index=False)
        suggestions_df.to_excel(writer, sheet_name='Suggested Parents', index=False)
        
        # Formatting
        workbook = writer.book
        format_header = workbook.add_format({'bold': True, 'bg_color': '#4472C4', 'font_color': 'white'})
        
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.autofilter(0, 0, 0, len(current_df.columns)-1)
            worksheet.freeze_panes(1, 0)
            
            # Apply header format
            for col_num, value in enumerate(current_df.columns.values):
                worksheet.write(0, col_num, value, format_header)
            
            # Auto-adjust columns' width
            for i, col in enumerate(current_df.columns):
                max_len = max((
                    current_df[col].astype(str).map(len).max(),
                    len(str(col))
                )) + 2
                worksheet.set_column(i, i, max_len)
    
    processed_data = output.getvalue()
    return processed_data

def main():
    st.title("Hierarchy Relationship Validator")
    
    # Initialize validator
    validator = ParentChildValidator('data/input.json', 'embeddings/embeddings.pkl')
    results = validator.validate_relationships()
    
    current_df, suggestions_df = prepare_data(results)
    
    # Display current relationships
    st.header("Current Relationships")
    st.dataframe(current_df)
    
    # Display all suggestions with similarity scores
    st.header("Suggested Parent Nodes")
    
    if suggestions_df.empty:
        st.info("No suggestions available")
    else:
        # Group by root and show suggestions
        for root_name, group in suggestions_df.groupby('Root Name'):
            current_score = current_df[current_df['Root Name'] == root_name]['Score'].values[0]
            
            with st.expander(f"{root_name} (Current Score: {current_score:.2f})"):
                st.dataframe(
                    group[['Suggested Parent', 'Similarity Score']]
                    .sort_values('Similarity Score', ascending=False)
                    .style.format({'Similarity Score': '{:.2f}'}),
                    use_container_width=True,
                    hide_index=True
                )
    
    # Add export button
    if st.button("📥 Export to Excel"):
        excel_data = to_excel(current_df, suggestions_df)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download Excel File",
            data=excel_data,
            file_name=f"hierarchy_validation_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()