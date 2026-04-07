import sys
print("The Python running this script is located at:")
print(sys.executable)

try:
    import langchain
    print(f"LangChain is installed! Version: {langchain.__version__}")
except ImportError:
    print("LangChain is STILL NOT installed in this specific Python path.")