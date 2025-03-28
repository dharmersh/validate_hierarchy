import streamlit as st
import pandas as pd
import os
from validator import ParentChildValidator  # Import your existing validator

# Set page config
st.set_page_config(
    page_title="Hierarchy Validator",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .st-emotion-cache-1v0mbdj {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-style {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .subheader-style {
        font-size: 18px;
        font-weight: bold;
        color: #3498db;
        margin-top: 20px;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load validation data using your existing validator"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'input.json')
    EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'embeddings', 'embeddings.pkl')
    
    validator = ParentChildValidator(
        data_path=DATA_PATH,
        embeddings_path=EMBEDDINGS_PATH
    )
    return validator.validate_relationships()

def prepare_report_data(results):
    """Prepare data for the Streamlit report"""
    old_relationships = []
    new_suggestions = []
    
    for result in results:
        # Old relationship data
        old_relationships.append({
            'Root Key': result['root_key'],
            'Root Name': result['root_name'],
            'Current Parent': result['current_parent']['parent_name'],
            'Similarity Score': result['current_parent']['similarity_score'],
            'Validation': result['validation']
        })
        
        # New suggestions data
        for suggestion in result['suggested_parents']:
            new_suggestions.append({
                'Root Key': result['root_key'],
                'Root Name': result['root_name'],
                'Current Parent': result['current_parent']['parent_name'],
                'Suggested Parent': suggestion['parent_name'],
                'Similarity Score': suggestion['similarity_score'],
                'Improvement': suggestion['similarity_score'] - result['current_parent']['similarity_score']
            })
    
    return pd.DataFrame(old_relationships), pd.DataFrame(new_suggestions)

def display_metrics(old_df, new_df):
    """Display summary metrics"""
    st.markdown('<div class="header-style">Validation Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Total Relationships", len(old_df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        passed = len(old_df[old_df['Validation'] == 'VALID'])
        st.metric("Valid Relationships", f"{passed} ({(passed/len(old_df))*100:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        avg_improvement = new_df['Improvement'].mean() if not new_df.empty else 0
        st.metric("Avg. Improvement Possible", f"{avg_improvement:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.title("Hierarchy Relationship Validator")
    st.markdown("Analyze and improve parent-child relationships in your hierarchy")
    
    # Load data
    with st.spinner("Loading and validating relationships..."):
        results = load_data()
        old_df, new_df = prepare_report_data(results)
    
    # Display metrics
    display_metrics(old_df, new_df)
    
    # Section 1: Current Relationships
    st.markdown('<div class="subheader-style">Current Relationships</div>', unsafe_allow_html=True)
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        validation_filter = st.selectbox(
            "Filter by Validation Status",
            ["All", "VALID", "INVALID"]
        )
    
    with col2:
        score_threshold = st.slider(
            "Minimum Similarity Score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )
    
    # Apply filters
    filtered_old = old_df.copy()
    if validation_filter != "All":
        filtered_old = filtered_old[filtered_old['Validation'] == validation_filter]
    filtered_old = filtered_old[filtered_old['Similarity Score'] >= score_threshold]
    
    # Display current relationships
    st.dataframe(
        filtered_old.sort_values('Similarity Score', ascending=False),
        use_container_width=True,
        height=400
    )
    
    # Section 2: Improvement Suggestions
    st.markdown('<div class="subheader-style">Improvement Suggestions</div>', unsafe_allow_html=True)
    
    if not new_df.empty:
        # Sort by improvement potential
        new_df = new_df.sort_values('Improvement', ascending=False)
        
        # Display suggestions
        st.dataframe(
            new_df,
            use_container_width=True,
            height=400
        )
        
        # Visualizations
        st.markdown('<div class="subheader-style">Improvement Analysis</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Score Distribution", "Top Improvements"])
        
        with tab1:
            st.bar_chart(new_df, x="Suggested Parent", y="Improvement")
        
        with tab2:
            top_improvements = new_df.head(10)
            st.bar_chart(top_improvements, x="Root Name", y=["Similarity Score", "Improvement"])
    else:
        st.info("No improvement suggestions available")

if __name__ == "__main__":
    main()