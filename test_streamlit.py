import streamlit as st
import pandas as pd
import os
import sys
from validator import ParentChildValidator

# Workaround for Streamlit-PyTorch conflict
if 'streamlit' in sys.modules:
    import torch
    torch.classes.__path__ = []  # Disable problematic PyTorch class inspection

# Set page config
st.set_page_config(
    page_title="Hierarchy Validator",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .header-style {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .root-container {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #4a90e2;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load validation data with error handling"""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_PATH = os.path.join(BASE_DIR, 'data', 'input.json')
        EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'embeddings', 'embeddings.pkl')
        
        validator = ParentChildValidator(
            data_path=DATA_PATH,
            embeddings_path=EMBEDDINGS_PATH
        )
        return validator.validate_relationships()
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return []

def prepare_data(results, min_score=0.6):
    """Prepare data with minimum score filtering"""
    current = []
    suggestions = []
    
    for result in results:
        # Current relationship
        current.append({
            'Root Key': result['root_key'],
            'Root Name': result['root_name'],
            'Current Parent': result['current_parent']['parent_name'],
            'Score': result['current_parent']['similarity_score'],
            'Status': result['validation']
        })
        
        # Suggested improvements (filtered by min_score)
        for suggestion in result['suggested_parents']:
            if suggestion['similarity_score'] >= min_score:
                suggestions.append({
                    'Root Key': result['root_key'],
                    'Root Name': result['root_name'],
                    'Current Parent': result['current_parent']['parent_name'],
                    'Current Score': result['current_parent']['similarity_score'],
                    'Suggested Parent': suggestion['parent_name'],
                    'New Score': suggestion['similarity_score'],
                    'Improvement': suggestion['similarity_score'] - result['current_parent']['similarity_score']
                })
    
    return pd.DataFrame(current), pd.DataFrame(suggestions)

def display_suggestions(suggestions_df):
    """Display all roots with their qualified suggestions"""
    st.markdown('<div class="header-style">Improvement Suggestions (Score > 60%)</div>', unsafe_allow_html=True)
    
    if suggestions_df.empty:
        st.info("No qualified improvement suggestions found (minimum 60% score)")
        return
    
    # Group by root to show all suggestions together
    grouped = suggestions_df.groupby('Root Name')
    
    for root_name, group in grouped:
        with st.container():
            st.markdown(f"""
            <div class="root-container">
                <h3>{root_name}</h3>
                <p><strong>Current Parent:</strong> {group['Current Parent'].iloc[0]} (Score: {group['Current Score'].iloc[0]:.2f})</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display all suggestions for this root
            st.dataframe(
                group[[
                    'Suggested Parent',
                    'New Score',
                    'Improvement'
                ]].sort_values('New Score', ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")

def main():
    st.title("Hierarchy Relationship Validator")
    
    # Load data with minimum 60% score filter
    with st.spinner("Analyzing relationships..."):
        results = load_data()
        if not results:
            st.stop()
        
        current_df, suggestions_df = prepare_data(results, min_score=0.6)
    
    # Metrics
    st.markdown('<div class="header-style">Validation Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Relationships", len(current_df))
    with col2:
        valid = len(current_df[current_df['Status'] == 'VALID'])
        st.metric("Valid", f"{valid} ({valid/len(current_df)*100:.1f}%)")
    with col3:
        qualified = len(suggestions_df['Root Name'].unique())
        st.metric("Roots with Improvements", qualified)
    
    # Current relationships
    st.markdown('<div class="header-style">Current Relationships</div>', unsafe_allow_html=True)
    
    # Display current relationships
    st.dataframe(
        current_df.sort_values('Score', ascending=False),
        use_container_width=True,
        height=400
    )
    
    # Improvement suggestions section
    display_suggestions(suggestions_df)

if __name__ == "__main__":
    main()
