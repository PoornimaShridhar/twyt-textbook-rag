import os
from utilities import Utilities
from chroma_collection_processor import ChromaCollection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import YoutubeLoader
import config
import vectorstore

class PromptProcessor:
    def __init__(self):
        # Initialize components only once in the constructor
        self.utilities = Utilities()
        self.collection_name = "combined_collection_recursive_sentence"
        self.results_folder = "results"
        # Configure the Groq chat model and Chroma client via `config`
        import llm_wrapper
        self.llm = llm_wrapper.get_groq_chat(temperature=0, model=getattr(config, "GROQ_MODEL", None))
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
        output = self.llm.invoke(full_prompt)
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

        import llm_wrapper
        llm = llm_wrapper.get_groq_chat(temperature=0, model=getattr(config, "GROQ_MODEL", None))

        # Prepare the prompt
        prompt_intro = (
            "You are a helpful YouTube assistant with expertise in video explanations. "
            "Please provide a concise summary highlighting the key points discussed."
        )
        prompt = f"Here are some documents: {page_content}. {prompt_intro}"

        result = llm.predict(prompt)

        return result.strip()

    except Exception as e:
        return f"An error occurred: {e}"

    
def webSearch(query):
    # Web search is intentionally disabled for now.
    # This placeholder stays here so the feature can be restored later.
    return {
        "links": [],
        "answer": (
            "Web search is currently commented out in the Groq-only setup. "
            "Re-enable this function later when you want web retrieval back."
        ),
    }
