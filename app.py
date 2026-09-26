import streamlit as st
import os
from utils.document_processor import extract_text_from_pdf, split_text_into_chunks
from utils.gemini_agent import analyze_document_risks, setup_rag_chain

st.set_page_config(page_title="Legal Assist", page_icon="⚖️", layout="wide")

# Custom CSS for a cleaner, professional look
st.markdown("""
<style>
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .risk-alert {
        padding: 1rem;
        background-color: #fff3cd;
        color: #856404;
        border-left: 5px solid #ffeeba;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .risk-label {
        font-weight: bold;
        color: #d9534f;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Legal Assist")
st.markdown("Upload a legal document (PDF or text) to automatically summarize clauses, identify risks, and ask questions.")

# Session State Initialization
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for Inputs
with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key. We'll check the GEMINI_API_KEY environment variable first if this is empty.")
    
    st.header("Document Input")
    input_method = st.radio("Choose Input Method", ("Upload PDF", "Paste Text"))
    
    doc_text = ""
    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file:
            doc_text = extract_text_from_pdf(uploaded_file)
    else:
        doc_text = st.text_area("Paste legal text here", height=200)

    analyze_btn = st.button("Analyze Document", type="primary")

# Resolve API Key
api_key = api_key_input or os.environ.get("GEMINI_API_KEY")

if analyze_btn:
    if not api_key:
        st.error("Please provide a Gemini API Key to proceed.")
    elif not doc_text.strip():
        st.error("Please provide a document (PDF or text).")
    else:
        with st.spinner("Analyzing document... This may take a moment."):
            st.session_state.document_text = doc_text
            
            # Risk Analysis
            st.session_state.analysis_results = analyze_document_risks(doc_text, api_key)
            
            # Setup RAG
            chunks = split_text_into_chunks(doc_text)
            st.session_state.rag_chain = setup_rag_chain(chunks, api_key)
            
            # Clear chat history on new document
            st.session_state.chat_history = []
            
        st.success("Analysis complete!")

# Main Area Layout
if st.session_state.analysis_results is not None:
    tab1, tab2 = st.tabs(["📄 Document Analysis", "💬 Q&A Chat"])
    
    with tab1:
        st.header("Clause Summaries & Risk Flags")
        for section in st.session_state.analysis_results:
            # Determine if this section has risks
            has_risks = len(section.get("risks", [])) > 0
            icon = "⚠️" if has_risks else "✅"
            
            with st.expander(f"{icon} {section.get('section_title', 'Section')}"):
                st.markdown(f"**Summary:** {section.get('summary', '')}")
                
                if has_risks:
                    for risk in section["risks"]:
                        st.markdown(f"""
                        <div class="risk-alert">
                            <span class="risk-label">WARNING: {risk.get('warning_label', 'Risk Found')}</span><br/>
                            <b>Clause:</b> <i>"{risk.get('clause_snippet', '')}"</i><br/>
                            <b>Explanation:</b> {risk.get('explanation', '')}
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Ask Questions about the Document")
        st.info("Ask follow-up questions. Answers are generated strictly based on the uploaded document text.")
        
        # Display chat messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message to state and display
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if st.session_state.rag_chain and api_key:
                        try:
                            response = st.session_state.rag_chain.invoke(prompt)
                            st.markdown(response)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Error querying the document: {e}")
                    else:
                        st.error("RAG chain not initialized or API key missing.")
else:
    st.info("Upload a document and click 'Analyze Document' to begin.")
