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
    .score-slider {
        margin-bottom: 30px;
    }
    .valid-card {
        border-left: 4px solid #2ecc71;
    }
    .invalid-card {
        border-left: 4px solid #e74c3c;
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

def prepare_data(results):
    """Prepare all data without filtering"""
    current = []
    suggestions = []
    
    for result in results:
        # Current relationship
        current.append({
            'Root Key': result['root_key'],
            'Root Name': result['root_name'],
            'Current Parent': result['current_parent']['parent_name'],
            'Score': result['current_parent']['similarity_score'],
            'Status': result['validation'],
            'Validation Status': result['validation_status']
        })
        
        # All suggested improvements
        for suggestion in result['suggested_parents']:
            suggestions.append({
                'Root Key': result['root_key'],
                'Root Name': result['root_name'],
                'Current Parent': result['current_parent']['parent_name'],
                'Current Score': result['current_parent']['similarity_score'],
                'Suggested Parent': suggestion['parent_name'],
                'New Score': suggestion['similarity_score'],
                'Improvement': suggestion['similarity_score'] - result['current_parent']['similarity_score'],
                'Status': 'IMPROVED' if suggestion['similarity_score'] > result['current_parent']['similarity_score'] else 'NOT IMPROVED'
            })
    
    return pd.DataFrame(current), pd.DataFrame(suggestions)

def filter_by_score(df, score_column, min_score):
    """Filter dataframe based on score threshold"""
    return df[df[score_column] >= min_score]

def display_current_relationships(current_df, validation_threshold):
    """Display current relationships with dynamic validation"""
    st.markdown(f'<div class="header-style">Current Relationships (Validation Threshold: ≥ {validation_threshold:.0%})</div>', unsafe_allow_html=True)
    
    # Apply dynamic validation based on threshold
    current_df['Dynamic Validation'] = current_df['Score'].apply(
        lambda x: 'VALID' if x >= validation_threshold else 'INVALID'
    )
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Relationships", len(current_df))
    with col2:
        valid = len(current_df[current_df['Dynamic Validation'] == 'VALID'])
        st.metric("Valid", f"{valid} ({valid/len(current_df)*100:.1f}%)")
    with col3:
        invalid = len(current_df[current_df['Dynamic Validation'] == 'INVALID'])
        st.metric("Invalid", f"{invalid} ({invalid/len(current_df)*100:.1f}%)")
    with col4:
        st.metric("Validation Threshold", f"{validation_threshold:.0%}")
    
    # Display current relationships with dynamic validation
    filtered_current = current_df.copy()
    
    # Filter options
    with st.expander("Filter Options", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            validation_filter = st.selectbox(
                "Filter by validation status",
                ["All", "VALID", "INVALID"],
                key="current_filter"
            )
        with col2:
            min_score = st.slider(
                "Minimum current score",
                0.0, 1.0, 0.0, 0.05,
                key="current_score_filter"
            )
    
    # Apply filters
    if validation_filter != "All":
        filtered_current = filtered_current[filtered_current['Dynamic Validation'] == validation_filter]
    filtered_current = filtered_current[filtered_current['Score'] >= min_score]
    
    # Display
    st.dataframe(
        filtered_current.sort_values('Score', ascending=False),
        use_container_width=True,
        height=400,
        column_config={
            "Dynamic Validation": st.column_config.TextColumn(
                "Validation",
                help="Dynamically calculated based on current threshold"
            )
        }
    )

def display_improvement_suggestions(suggestions_df, improvement_threshold):
    """Display improvement suggestions with dynamic filtering"""
    st.markdown(f'<div class="header-style">Improvement Suggestions (Score ≥ {improvement_threshold:.0%})</div>', unsafe_allow_html=True)
    
    # Filter suggestions
    filtered_suggestions = filter_by_score(suggestions_df, 'New Score', improvement_threshold)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Suggestions", len(filtered_suggestions))
    with col2:
        improved = len(filtered_suggestions[filtered_suggestions['Status'] == 'IMPROVED'])
        st.metric("Improving", improved)
    with col3:
        st.metric("Minimum Score", f"{improvement_threshold:.0%}")
    
    # Threshold adjustment
    with st.expander("Adjust Thresholds", expanded=True):
        new_threshold = st.slider(
            "Set minimum suggestion score:",
            0.0, 1.0, improvement_threshold, 0.05,
            key="improvement_threshold",
            format="%.2f"
        )
        
        min_improvement = st.slider(
            "Minimum improvement required:",
            0.0, 1.0, 0.0, 0.05,
            key="min_improvement",
            format="%.2f"
        )
    
    # Apply additional filtering
    filtered_suggestions = filtered_suggestions[
        (filtered_suggestions['New Score'] >= new_threshold) &
        (filtered_suggestions['Improvement'] >= min_improvement)
    ]
    
    if filtered_suggestions.empty:
        st.info(f"No suggestions meet the current thresholds (Score ≥ {new_threshold:.0%}, Improvement ≥ {min_improvement:.2f})")
        return
    
    # Group by root
    grouped = filtered_suggestions.groupby('Root Name')
    
    for root_name, group in grouped:
        card_class = "valid-card" if group['Current Score'].iloc[0] >= new_threshold else "invalid-card"
        
        with st.container():
            st.markdown(f"""
            <div class="root-container {card_class}">
                <h3>{root_name}</h3>
                <p><strong>Current:</strong> {group['Current Parent'].iloc[0]} (Score: {group['Current Score'].iloc[0]:.2f})</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display suggestions
            st.dataframe(
                group[[
                    'Suggested Parent',
                    'New Score',
                    'Improvement',
                    'Status'
                ]].sort_values('New Score', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "New Score": st.column_config.NumberColumn(format="%.2f"),
                    "Improvement": st.column_config.NumberColumn(format="+%.2f")
                }
            )
            
            st.markdown("---")

def main():
    st.title("Hierarchy Relationship Validator")
    
    # Initialize session state for thresholds
    if 'validation_threshold' not in st.session_state:
        st.session_state.validation_threshold = 0.6
    if 'improvement_threshold' not in st.session_state:
        st.session_state.improvement_threshold = 0.6
    
    # Load all data
    with st.spinner("Analyzing relationships..."):
        results = load_data()
        if not results:
            st.stop()
        
        current_df, all_suggestions_df = prepare_data(results)
    
    # Global threshold controls
    with st.sidebar:
        st.markdown("### Global Threshold Settings")
        st.session_state.validation_threshold = st.slider(
            "Validation Threshold",
            0.0, 1.0, st.session_state.validation_threshold, 0.05,
            key="global_validation_threshold"
        )
        st.session_state.improvement_threshold = st.slider(
            "Improvement Threshold",
            0.0, 1.0, st.session_state.improvement_threshold, 0.05,
            key="global_improvement_threshold"
        )
    
    # Current relationships section
    display_current_relationships(
        current_df,
        st.session_state.validation_threshold
    )
    
    # Improvement suggestions section
    display_improvement_suggestions(
        all_suggestions_df,
        st.session_state.improvement_threshold
    )

if __name__ == "__main__":
    main()