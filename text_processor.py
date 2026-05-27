from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.llms import Ollama


class TextProcessor:
    def __init__(self):
        """
        Initialize the text processing class.
        """
        pass
    
    def split_text(self, formatted_texts, splitter):
        """
        Split text using the provided splitter (character-based or token-based).
        """
        chunks_with_metadata = []
        for entry in formatted_texts:
            text = entry['text']
            metadata = entry['metadata']
            split_texts = splitter.split_text(text)
            for chunk in split_texts:
                chunks_with_metadata.append({'metadata': metadata, 'text': chunk})
        return chunks_with_metadata

    def prepare_documents_and_metadata(self, final_chunks, include_metadata=True):
        """
        Prepare documents and metadata for Chroma collection.
        """
        documents = [chunk['text'] for chunk in final_chunks]

        if include_metadata:
            metadata = []
            for chunk in final_chunks:
                metadata_dict = {}
                for line in chunk['metadata'].split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata_dict[key.strip()] = value.strip()
                metadata.append(metadata_dict)
        else:
            metadata = None
        return documents, metadata
