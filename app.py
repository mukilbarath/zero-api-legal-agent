import streamlit as st
import tempfile
import os
import warnings
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore")

# --- UI Configuration ---
st.set_page_config(page_title="AI Legal Analyst", page_icon="⚖️", layout="wide")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "assessment" not in st.session_state:
    st.session_state.assessment = None

# --- Core AI Functions ---
def process_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    unique_db_dir = f"./chroma_temp_{uuid.uuid4().hex}"
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=unique_db_dir
    )
    return vector_db, chunks

def generate_initial_assessment(chunks, jurisdiction):
    llm = Ollama(model="llama3")
    sample_text = "\n\n".join([doc.page_content for doc in chunks[:4]])
    
    prompt = f"""You are an expert corporate lawyer practicing in {jurisdiction}. 
    Review the following excerpts from a newly uploaded legal document. 
    Evaluate this strictly according to the constitution, corporate law, and legal precedent of {jurisdiction}.
    Provide a structured, rapid-fire assessment with these exact three sections. Be concise.

    1. DOCUMENT COMPLEXITY: (Rate from 1/10 to 10/10 and briefly explain why)
    2. RISK SCALE: (State 'Low', 'Medium', or 'High', and identify the biggest potential legal risk)
    3. KEY CONCENTRATION AREAS: (List 3 short bullet points the user MUST pay attention to)

    Document Excerpts:
    {sample_text}
    """
    
    with st.spinner(f"Generating automated risk assessment under {jurisdiction} jurisdiction..."):
        return llm.invoke(prompt)

# 👉 FEATURE 1 LIVES HERE: Dynamic Prompt Injection based on the toggle
def ask_question(question, vector_db, jurisdiction, plain_english):
    llm = Ollama(model="llama3")
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    if plain_english:
        system_prompt = (
            f"You are a helpful legal assistant in {jurisdiction}. "
            "Use the retrieved context to answer the user's question. "
            "CRITICAL INSTRUCTION: You must explain the answer in extremely simple, plain English. "
            "Use everyday analogies. Do not use ANY dense legal jargon. Explain it like the user is a beginner. "
            "If the answer is not in the context, explicitly state that you cannot find it."
            "\n\nContext:\n{context}" # <--- THE FIX: Single braces
        )
    else:
        system_prompt = (
            f"You are a meticulous, highly experienced corporate lawyer and legal analyst in {jurisdiction}. "
            f"You must evaluate the user's question strictly according to the constitution and legal frameworks of {jurisdiction}. "
            "Use the following pieces of retrieved legal context to answer the user's question. "
            "If the answer is not in the context, explicitly state that you cannot find it. "
            "\n\nContext:\n{context}" # <--- THE FIX: Single braces
        )
        
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain.invoke({"input": question})
# 👉 FEATURE 3 LIVES HERE: The Redrafting Engine
def redraft_clause(clause, favor_party, jurisdiction):
    llm = Ollama(model="llama3")
    prompt = f"""You are a ruthless, highly skilled contract lawyer in {jurisdiction}. 
    Your client is the {favor_party}. 
    Rewrite the following legal clause so that it heavily protects and favors the {favor_party}, while still remaining legally enforceable.
    Output ONLY the newly drafted clause, followed by a brief bulleted list explaining what you changed to protect your client.
    
    Original Clause:
    {clause}
    """
    with st.spinner(f"Drafting aggressive new clause favoring the {favor_party}..."):
        return llm.invoke(prompt)

# ==========================================
# --- APP LAYOUT & UI ---
# ==========================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    selected_country = st.selectbox(
        "Select Legal Jurisdiction:",
        ("India", "United States", "United Kingdom", "European Union", "Singapore")
    )
    
    # 👉 FEATURE 1 UI: The Toggle
    st.markdown("---")
    st.subheader("🧠 AI Persona")
    plain_english_toggle = st.toggle("Explain in Plain English", value=False, help="Strips away legal jargon and uses simple analogies.")
    
    st.markdown("---")
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF Contract", type="pdf")
    
    if uploaded_file is not None and st.session_state.vector_db is None:
        st.info("Ingesting document...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        vector_db, chunks = process_document(tmp_path)
        st.session_state.vector_db = vector_db
        st.session_state.assessment = generate_initial_assessment(chunks, selected_country)
        
        os.remove(tmp_path)
        st.success("Document processed successfully!")
        st.rerun()
        
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

st.title("⚖️ Zero-API Legal Analyst")

if st.session_state.vector_db is None:
    st.markdown("### 👈 Configure your jurisdiction and upload a legal document to begin.")
else:
    # Set up the two workspace tabs
    tab1, tab2 = st.tabs(["💬 Analysis & Chat", "✍️ Clause Redrafting Studio"])
    
    # ----------------------------------------
    # TAB 1: The original Chat & Assessment
    # ----------------------------------------
    with tab1:
        st.subheader(f"📊 Automated Document Profiling ({selected_country} Law)")
        st.info(st.session_state.assessment)
        st.markdown("---")
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        user_query = st.chat_input(f"Ask a question regarding {selected_country} law...")
        
        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing clauses..."):
                    # Pass the toggle state into the engine
                    response = ask_question(user_query, st.session_state.vector_db, selected_country, plain_english_toggle)
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    with st.expander("🔍 View Sourced Legal Clauses"):
                        for i, doc in enumerate(response["context"]):
                            st.write(f"**Excerpt {i+1}:** {doc.page_content.strip()}")
                            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # ----------------------------------------
    # 👉 FEATURE 3 UI: TAB 2
    # ----------------------------------------
    with tab2:
        st.subheader("Aggressive Redrafting Engine")
        st.markdown("Found a high-risk clause in the chat? Paste it here and have the AI rewrite it to protect your interests.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            target_clause = st.text_area("Paste the original risky clause here:", height=200)
        
        with col2:
            st.markdown("**Drafting Strategy**")
            favor_party = st.text_input("Who should this favor?", placeholder="e.g., Tenant, Founder, Buyer")
            redraft_btn = st.button("Rewrite Clause", type="primary", use_container_width=True)
            
        if redraft_btn:
            if target_clause and favor_party:
                new_clause = redraft_clause(target_clause, favor_party, selected_country)
                st.success("Drafting Complete!")
                st.markdown("### 🛡️ Your New Clause:")
                st.write(new_clause)
            else:
                st.warning("Please paste a clause and specify who it should favor.")