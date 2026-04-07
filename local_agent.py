import warnings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
# --- THE FIX: Using the new langchain_classic namespace ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# ----------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate

# Suppress warnings for clean terminal output
warnings.filterwarnings("ignore")

def ask_legal_agent(question):
    print("Waking up the Legal Analyst Agent...\n")

    # 1. Connect to the existing local Vector DB
    print("1. Connecting to local ChromaDB memory...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db_directory = "./chroma_db"
    
    # Load the database you just created
    vector_db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    
    # Set up the retriever to fetch the 3 most relevant chunks of the contract
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 2. Initialize the Local LLM (Llama-3)
    print("2. Loading local Llama-3 brain...")
    llm = Ollama(model="llama3")

    # 3. Create the System Prompt (The Agent's Persona)
    system_prompt = (
        "You are a meticulous, highly experienced corporate lawyer and legal analyst. "
        "Use the following pieces of retrieved legal context to answer the user's question. "
        "If the answer is not in the context, explicitly state that you cannot find it in the provided document. "
        "Do not invent or hallucinate legal clauses. "
        "\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Build the RAG Chain
    print("3. Searching documents and analyzing...\n")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 5. Ask the Question
    response = rag_chain.invoke({"input": question})
    
    print("========================================")
    print("👩‍⚖️ LEGAL ANALYST RESPONSE:")
    print("========================================")
    print(response["answer"])
    print("========================================")
    
    # Print out the exact sources it used
    print("\n🔍 SOURCES CITED FROM DATABASE:")
    for i, doc in enumerate(response["context"]):
        print(f"\n--- Excerpt {i+1} ---")
        print(doc.page_content.strip())

if __name__ == "__main__":
    # Change this question to match whatever text is actually inside your sample_doc.pdf
    user_question = "What does the document say about payment, rent, or termination?"
    ask_legal_agent(user_question)