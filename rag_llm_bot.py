import os
from utilities import Utilities
from chroma_collection_processor import ChromaCollection
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.llms import Ollama
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
import config
import vectorstore

class PromptProcessor:
    def __init__(self):
        # Initialize components only once in the constructor
        self.utilities = Utilities()
        self.collection_name = "combined_collection_recursive_sentence"
        self.results_folder = "results"
        # Configure Ollama and Chroma client via `config`
        # Use llm_wrapper to create Ollama instance
        import llm_wrapper
        ollama_base = config.OLLAMA_BASE_URL or "http://128.105.144.59:11434"
        ollama_model = config.OLLAMA_MODEL or "llama3"
        self.llm = llm_wrapper.get_ollama(model=ollama_model, base_url=ollama_base)
        # Use centralized vectorstore client
        self.client = vectorstore.get_client()
        self.embedding_function = SentenceTransformerEmbeddingFunction()
        # Get collection only once (use chromadb client directly)
        self.chroma_collection = self.client.get_collection(name=self.collection_name, embedding_function=self.embedding_function)
        
    def process_prompt(self,prompt, conversation_history=None):
        if conversation_history:
            full_prompt = f"{conversation_history}\nUser: {prompt}"
        else:
            full_prompt = f"User: {prompt}"
        query = prompt.strip().split('. ', 1)[-1]
        results = self.chroma_collection.query(query_texts=[query], n_results=5)
        retrieved_documents = results['documents'][0]
        retrieved_metadatas = results['metadatas'][0]
        self.utilities.print_retrieved_results(query, retrieved_documents, retrieved_metadatas)
        processed_metadata = self.utilities.process_metadata(retrieved_metadatas)
        print(processed_metadata)
        output = self.llm.invoke(full_prompt) if isinstance(self.llm, Ollama) else self.llm.query_llm(query=query, retrieved_documents=retrieved_documents)
        retrieved_data = {
            "metadata": processed_metadata,
            "output": output
        }
        result_file_path = os.path.join(self.results_folder, f'result.json')
        self.utilities.save_to_file(retrieved_data, result_file_path)
        return retrieved_data


def ytloader(video_url): 
    print("========================================================================================")
    try:
        loader = YoutubeLoader.from_youtube_url(
            video_url, add_video_info=False
        )
        documents = loader.load()
        if not documents:
            return "No documents found for the provided video URL."
        page_content = documents[0].page_content

        # Initialize ChatOpenAI model via wrapper
        import llm_wrapper
        llm = llm_wrapper.get_chat_openai(temperature=0, model="gpt-4o")
        # memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

        # Prepare the prompt
        prompt_intro = (
            "You are a helpful YouTube assistant with expertise in video explanations. "
            "Please provide a concise summary highlighting the key points discussed."
        )
        prompt = f"Here are some documents: {page_content}. {prompt_intro}"

        # Invoke the GPT model
        result = llm.predict(prompt)
        # memory.save_context({"input": prompt}, {"output": result})

        return result.strip()

    except Exception as e:
        return f"An error occurred: {e}"

    
def webSearch(query):
    if getattr(config, 'TAVILY_API_KEY', None):
        os.environ["TAVILY_API_KEY"] = config.TAVILY_API_KEY
    tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    )
    response = tool.invoke({"query": query})
    links = []
    for result in response:
        if isinstance(result, dict) and 'url' in result:
            links.append(result['url'])


    prompt = f"""
        You are  a helpful content summarizer. Based on the given information, answer the question :{query}
        Here is the relevant information:{response}
        """

    import llm_wrapper
    llm = llm_wrapper.get_chat_openai(temperature=0, model="gpt-4o")
    result = llm_wrapper.invoke_llm(llm, prompt)

    results_folder = "web_search_results"
    utilities = Utilities()
    result_file_path = os.path.join(results_folder, f'result.json')
    utilities.save_to_file(result, result_file_path)
    return {"links": links, "answer": result}
