import streamlit as st
import sys
import os
import base64



sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from engine import get_llm, create_query_engine
except ImportError as e:
    st.error(f" Import Error: {e}. Please make sure 'engine.py' is in the same folder as 'app.py'.")
    st.stop()


@st.cache_data
def get_svg_as_base64(file_path):
    """Loads an SVG file, base64 encodes it, and returns a data URI."""
    try:
        with open(file_path, "rb") as f:
            svg_bytes = f.read()
        b64_svg = base64.b64encode(svg_bytes).decode("utf-8")
        return f"data:image/svg+xml;base64,{b64_svg}"
    except FileNotFoundError:
        return None


ICON_FILE = "natroadwhite.svg"
icon_data_uri = get_svg_as_base64(ICON_FILE)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Road Safety Intervention GPT",
    page_icon=icon_data_uri if icon_data_uri else "🛣️",  # Use SVG if found, else fallback
    layout="wide"
)


def load_css(file_name):
    """A function to inject a local CSS file."""
    try:
        with open(file_name) as f:
           
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f" Warning: {file_name} not found. Styles may not apply.")


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


load_css("style.css")



google_api_key = st.secrets.get("GOOGLE_API_KEY", None) 
llm = get_llm(google_api_key)

if google_api_key:
    model_mode = "Cloud Mode (Gemini 1.5 Pro)"
else:
    model_mode = "Local Mode (Llama 3 8B)"


query_engine = load_rag_engine(llm)




if icon_data_uri:
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: left;">
        <img src="{icon_data_uri}" alt="Icon" style="width: 40px; height: 40px; margin-right: 10px;">
        <h1 style="margin: 0; color: white;">Road Safety Intervention GPT</h1>
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("Road Safety Intervention GPT")

st.markdown("Ask a question about road safety interventions from the provided database.")


with st.container():
    st.info(f"Running in: **{model_mode}**")
    user_query = st.text_input("Enter your query:", 
                               placeholder="e.g., 'What are the rules for a STOP sign?'",
                               label_visibility="collapsed")
    
    submit_button = st.button("Generate Answer")


output_container = st.container()

if submit_button:
    if not user_query:
        output_container.warning("Please enter a query.")
    else:
        with st.spinner("Searching database and generating answer..."):
            try:
               
                
                response = query_engine.query(user_query)
                
              
                output_container.success("Answer generated successfully!")
                output_container.markdown(str(response.response))
                
              
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