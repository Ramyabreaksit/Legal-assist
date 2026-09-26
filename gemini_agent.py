import os
import json
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Globals to hold the vector store and retriever to prevent re-initialization if not needed
_vectorstore = None

def get_gemini_api_key():
    return os.environ.get("GEMINI_API_KEY")

def configure_gemini(api_key: str):
    """Configures the Gemini SDK."""
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY"] = api_key # Needed for Langchain

def analyze_document_risks(text: str, api_key: str) -> list:
    """
    Uses Gemini to analyze the entire text (or a substantial part of it)
    and extract summaries and risky clauses.
    We ask for JSON output so we can reliably render it in Streamlit.
    """
    configure_gemini(api_key)
    # Using Gemini 1.5 Pro or Flash for large context. Flash is faster for this.
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are an expert legal assistant. Review the following legal document and break it down into major sections.
    For each section, provide a plain-language summary. 
    Crucially, flag any potentially risky or unusual clauses (e.g., auto-renewal, hidden fees, one-sided liability, forced arbitration, unreasonable data usage).
    
    Return the output ONLY as a valid JSON list of objects, without any markdown formatting like ```json.
    Each object should have the following schema:
    {{
        "section_title": "Name of the section",
        "summary": "Plain language summary of the section",
        "risks": [
            {{
                "clause_snippet": "A short snippet or description of the specific risky clause",
                "warning_label": "Short label (e.g., Auto-Renewal, Hidden Fee)",
                "explanation": "Why this is risky for the user"
            }}
        ] // Leave empty if no risks found in this section
    }}
    
    Document Text:
    {text}
    """
    
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        # Clean up if the model includes markdown formatting
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
             content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        return json.loads(content.strip())
    except Exception as e:
        print(f"Error during risk analysis: {{e}}")
        return []

def setup_rag_chain(chunks: list[str], api_key: str):
    """
    Initializes ChromaDB with the given chunks and Gemini embeddings.
    Returns a LangChain runnable chain for QA.
    """
    global _vectorstore
    
    configure_gemini(api_key)
    
    # Use Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Create or update Chroma vector store (in-memory for this session)
    _vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Setup LLM for chat
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    template = """
    You are a helpful legal assistant. Answer the user's question based STRICTLY on the context provided from the uploaded document.
    If the answer cannot be found in the context, state clearly that the document does not contain this information. 
    Do not hallucinate or use outside legal knowledge to make assumptions.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\\n\\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
