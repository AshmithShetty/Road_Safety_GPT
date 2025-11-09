import streamlit as st
import sys
import os

# PATH SETUP

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from engine import get_llm, create_query_engine
except ImportError as e:
    st.error(f" Import Error: {e}. Please make sure 'engine.py' is in the same folder as 'app.py'.")
    st.stop()

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Road Safety Intervention GPT",
    page_icon="🛣️",
    layout="wide"
)


def load_css(file_name):
    """A function to inject a local CSS file."""
    try:
        with open(file_name) as f:
           
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f" Warning: {file_name} not found. Styles may not apply.")

#CACHING
@st.cache_resource
def load_rag_engine(_llm):
    """
    Loads the RAG query engine from engine.py.
    We use @st.cache_resource to ensure this function runs only ONCE.
    Note: '_llm' underscore prefix tells Streamlit not to hash this argument.
    """
    try:
        return create_query_engine(_llm)
    except Exception as e:
        st.error(f"Error loading RAG engine: {e}")
        st.stop()

#MAIN APP LOGIC

# 1. LOAD CSS
load_css("style.css")

# 2. SET UP HYBRID SWITCH

google_api_key = st.secrets.get("GOOGLE_API_KEY", None) 
llm = get_llm(google_api_key)

if google_api_key:
    model_mode = "Cloud Mode (Gemini 1.5 Pro)"
else:
    model_mode = "Local Mode (Llama 3 8B)"

# 3. LOAD QUERY ENGINE
query_engine = load_rag_engine(llm)

# 4. UI
st.title("🛣️ National Road Safety Intervention GPT")
st.markdown("Ask a question about road safety interventions from the provided database.")

# Input Section
with st.container():
    st.info(f"Running in: **{model_mode}**")
    user_query = st.text_input("Enter your query:", 
                               placeholder="e.g., 'What are the rules for a STOP sign?'",
                               label_visibility="collapsed")
    
    submit_button = st.button("Generate Answer")

# Output Section
output_container = st.container()

if submit_button:
    if not user_query:
        output_container.warning("Please enter a query.")
    else:
        with st.spinner("Searching database and generating answer..."):
            try:
                # 5. RUN QUERY
                
                response = query_engine.query(user_query)
                
                # 6. DISPLAY RESPONSE
                output_container.success("Answer generated successfully!")
                output_container.markdown(str(response.response))
                
                # 7. DISPLAY SOURCES
                with output_container.expander("Show Sources Used"):
                    if response.source_nodes:
                        for i, node in enumerate(response.source_nodes):
                            st.subheader(f"Source {i+1} (Relevance: {node.score:.2f})")
                            st.markdown(f"**Code:** {node.metadata.get('code', 'N/A')}")
                            st.markdown(f"**Clause:** {node.metadata.get('clause', 'N/A')}")
                            st.markdown(f"**Problem:** {node.metadata.get('problem', 'N/A')}")
                            st.markdown(f"**Type:** {node.metadata.get('type', 'N/A')}")
                            st.divider()
                            st.markdown(f"**Text:**\n{node.get_content()}")
                    else:
                        st.warning("No sources were found for this query.")
            
            except Exception as e:
                output_container.error(f"An error occurred during generation: {e}")