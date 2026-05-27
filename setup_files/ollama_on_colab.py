!pip install langchain langchain_community langchain_core
from langchain_community.llms import Ollama

ollama_llm = Ollama( model="llama3", base_url="http://128.105.144.60:11434" )

response = ollama_llm.invoke("Explain different weather types")
