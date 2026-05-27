"""Vectorstore abstraction for creating Chroma client, vectorstore, and retrievers.

This module centralizes creation of PersistentClient, Chroma vectorstores,
and helpers to build retrievers. It keeps code paths consistent across modules.
"""
from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import config


def get_client(path=None):
    """Return a chromadb PersistentClient for the given path or from config."""
    client_path = path or getattr(config, 'CHROMA_CLIENT_PATH', None) or 'Collections'
    return PersistentClient(path=client_path)


def create_vectorstore(collection_name=None, client=None, embedding_function=None):
    """Create and return a (vectorstore, client) tuple.

    - `collection_name` defaults to config.DEFAULT_COLLECTION_NAME
    - `client` defaults to PersistentClient created by `get_client()`
    - `embedding_function` defaults to `OpenAIEmbeddings()`
    """
    client = client or get_client()
    collection_name = collection_name or getattr(config, 'DEFAULT_COLLECTION_NAME', 'text_Collection_3')
    embedding_function = embedding_function or OpenAIEmbeddings()
    vs = Chroma(collection_name=collection_name, embedding_function=embedding_function, client=client)
    return vs, client


def make_multi_vector_retriever(vectorstore, docstore=None, id_key='doc_id', k=5):
    """Return a MultiVectorRetriever for the given vectorstore.

    This defers import of langchain retriever classes to avoid heavy imports
    at module import time.
    """
    from langchain.storage import InMemoryStore
    from langchain.retrievers.multi_vector import MultiVectorRetriever

    docstore = docstore or InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
        search_kwargs={"k": k},
    )
    return retriever
