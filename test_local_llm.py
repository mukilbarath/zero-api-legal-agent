from langchain_community.llms import Ollama

def test_local_agent():
    print("Connecting to local Llama-3...")
    
    # Initialize the local model via Ollama
    llm = Ollama(model="llama3")
    
    # A test prompt fitting your legal use case
    prompt = """
    You are a meticulous legal editor. 
    Correct any obvious spelling mistakes in the following messy OCR text extracted from a handwritten deed.
    Do not add new information, just fix the spelling.
    
    Text: "The propperty is located at 123 Main strit. The tennant agrees to pay rent on the furst of every month."
    """
    
    print("Processing prompt locally (no internet required)...")
    response = llm.invoke(prompt)
    
    print("\n--- Output ---")
    print(response)

if __name__ == "__main__":
    test_local_agent()