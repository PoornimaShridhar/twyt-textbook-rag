"""Wrapper around Chroma collections used by the project.

Provides a simple class `ChromaCollection` to create/get collections,
add/query documents, and save/load embeddings. Uses the centralized
`vectorstore` helper for client creation where possible.
"""

import json
import chromadb
import numpy as np
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
import vectorstore

class ChromaCollection:
    MAX_BATCH_SIZE = 5000

    def __init__(self, collection_name):
        """
        Initialize the Chroma Collection with a collection name.
        """
        self.collection_name = collection_name
        self.collection = self.create_chroma_collection()

    def create_chroma_collection(self):
        """
        Create or retrieve a Chroma collection.
        """
        embedding_function = SentenceTransformerEmbeddingFunction()
        client = vectorstore.get_client()
        chroma_collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        return chroma_collection

    def add_to_chroma_collection(self, documents, metadata):
        """
        Add documents and metadata to the Chroma collection.
        """
        for start_index in range(0, len(documents), self.MAX_BATCH_SIZE):
            end_index = start_index + self.MAX_BATCH_SIZE
            batch_documents = documents[start_index:end_index]
            batch_ids = [str(i) for i in range(start_index, min(end_index, len(documents)))]

            if metadata:
                batch_metadata = metadata[start_index:end_index]
                self.collection.add(ids=batch_ids, documents=batch_documents, metadatas=batch_metadata)
            else:
                self.collection.add(ids=batch_ids, documents=batch_documents)

    def query_chroma_collection(self, query, n_results=5):
        """
        Perform a query on the Chroma collection and retrieve documents and metadata.
        """
        results = self.collection.query(query_texts=[query], n_results=n_results)
        retrieved_documents = results['documents'][0]
        retrieved_metadata = results['metadatas'][0]
        return retrieved_documents, retrieved_metadata

    def apply_metadata_filters(self, docs, metas, filters):
        """
        Filter documents and metadata based on specified filter criteria.
        """
        filtered_docs = []
        filtered_metas = []
        for doc, meta in zip(docs, metas):
            if all(meta.get(key) == value for key, value in filters.items()):
                filtered_docs.append(doc)
                filtered_metas.append(meta)
        return filtered_docs, filtered_metas

    def save_embeddings_to_file(self, file_path):
        """
        Save embeddings to a file.
        """
        # Fetch embeddings from the collection
        embeddings_data = self.collection.get(include=['embeddings'])
        embeddings = embeddings_data['embeddings']

        # Save embeddings to a JSON file
        with open(file_path, 'w') as file:
            json.dump(embeddings, file)

    def load_embeddings_from_file(self, file_path):
        """
        Load embeddings from a file.
        """
        with open(file_path, 'r') as file:
            embeddings_list = json.load(file)
        
        # Convert list to numpy array
        embeddings = np.array(embeddings_list)
        return embeddings
