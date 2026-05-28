"""Vectorstore abstraction for creating Chroma client, vectorstore, and retrievers.

This module centralizes creation of PersistentClient, Chroma vectorstores,
and helpers to build retrievers. It keeps code paths consistent across modules.
"""
from chromadb import PersistentClient
from langchain_chroma import Chroma
import config
from sentence_transformers import SentenceTransformer

# Provide a resilient InMemoryStore symbol: try known packages, otherwise a lightweight fallback
try:
    # Preferred recent community package
    from langchain_community.storage import InMemoryStore  # type: ignore
except Exception:
    try:
        # Older single-package LangChain
        from langchain.storage import InMemoryStore  # type: ignore
    except Exception:
        class InMemoryStore:
            """Minimal fallback docstore implementation used when LangChain's
            InMemoryStore isn't available. Implements the small subset of the
            API used by this project (mset, mget, get, and set operations).
            """
            def __init__(self):
                self._store = {}

            def mset(self, items):
                for k, v in items:
                    self._store[k] = v

            def mget(self, keys):
                return [self._store.get(k) for k in keys]

            def get(self, key, default=None):
                return self._store.get(key, default)

            def set(self, key, value):
                self._store[key] = value

# ConversationBufferMemory compatibility shim
try:
    from langchain.memory import ConversationBufferMemory  # type: ignore
except Exception:
    try:
        from langchain_community.memory import ConversationBufferMemory  # type: ignore
    except Exception:
        class ConversationBufferMemory:
            """Minimal ConversationBufferMemory fallback implementing the
            small API surface used in this project: `load_memory_variables`
            and `save_context`.
            """
            def __init__(self, return_messages=True, memory_key="chat_history"):
                self.return_messages = return_messages
                self.memory_key = memory_key
                self._history = []

            def load_memory_variables(self, inputs=None):
                # Return a dict matching LangChain's memory API
                return {self.memory_key: list(self._history)}

            def save_context(self, inputs, outputs):
                # Append a simple record; inputs/outputs are dicts
                self._history.append({"input": inputs, "output": outputs})

# MultiVectorRetriever compatibility shim
try:
    from langchain.retrievers.multi_vector import MultiVectorRetriever  # type: ignore
except Exception:
    class MultiVectorRetriever:
        """Minimal retriever fallback used when LangChain's retriever package
        is unavailable.

        It supports the small surface used in this project: construction with a
        vectorstore/docstore/id_key/search_kwargs tuple and `invoke(query)`.
        """

        def __init__(self, vectorstore, docstore=None, id_key="doc_id", search_kwargs=None):
            self.vectorstore = vectorstore
            self.docstore = docstore or InMemoryStore()
            self.id_key = id_key
            self.search_kwargs = search_kwargs or {"k": 5}

        def invoke(self, query):
            k = self.search_kwargs.get("k", 5)
            return self.vectorstore.similarity_search(query, k=k)

        def get_relevant_documents(self, query):
            return self.invoke(query)


class SentenceTransformerEmbeddings:
    """Lightweight embeddings adapter backed by sentence-transformers.

    Chroma/LangChain expect flat float vectors. This adapter keeps the output
    shape explicit and avoids nested lists from alternate embedding shims.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text):
        vector = self.model.encode([text], convert_to_numpy=True)[0]
        return vector.tolist()



def get_client(path=None):
    """Return a chromadb PersistentClient for the given path or from config."""
    client_path = path or getattr(config, 'CHROMA_CLIENT_PATH', None) or 'Collections'
    client = PersistentClient(path=client_path)
    try:
        print(f"[vectorstore.get_client] PersistentClient path={client_path} client={client}")
    except Exception:
        pass
    return client


def create_vectorstore(collection_name=None, client=None, embedding_function=None):
    """Create and return a (vectorstore, client) tuple.

    - `collection_name` defaults to config.DEFAULT_COLLECTION_NAME
    - `client` defaults to PersistentClient created by `get_client()`
    - `embedding_function` defaults to a SentenceTransformer embedding function
    """
    client = client or get_client()
    collection_name = collection_name or getattr(config, 'DEFAULT_COLLECTION_NAME', 'text_Collection_3')
    embedding_function = embedding_function or SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # If the configured/default collection is empty, try to fall back to
    # the first non-empty collection in the Chromadb client. This helps when
    # ingestion created collections with different names (chunking strategies).
    try:
        print(f"[vectorstore.create_vectorstore] requested default collection='{collection_name}'")
        collections = client.list_collections()
        try:
            print(f"[vectorstore.create_vectorstore] raw collections list: {collections}")
        except Exception:
            pass
        names = []
        for c in collections:
            if isinstance(c, dict) and 'name' in c:
                names.append(c['name'])
            else:
                try:
                    names.append(getattr(c, 'name'))
                except Exception:
                    continue

        chosen = collection_name if collection_name in names else None
        try:
            print(f"[vectorstore.create_vectorstore] discovered collection names: {names}")
        except Exception:
            pass

        # If the requested collection exists, verify it contains documents using a robust get()
        def collection_size(coll_name):
            try:
                coll = client.get_collection(coll_name)
                # First try Coll.count() which some chroma clients expose
                try:
                    cnt = coll.count()
                    return int(cnt)
                except Exception:
                    # Fallback: inspect returned data
                    try:
                        data = coll.get(include=['documents','metadatas','ids'])
                    except Exception:
                        data = coll.get(include=['documents','metadatas'])
                    for key in ('documents','metadatas','ids'):
                        val = data.get(key)
                        if val:
                            return len(val[0]) if isinstance(val[0], list) else len(val)
                    return 0
            except Exception:
                return 0

        if chosen:
            try:
                if collection_size(chosen) == 0:
                    chosen = None
            except Exception:
                try:
                    print(f"[vectorstore.create_vectorstore] error while checking size of chosen '{chosen}'")
                except Exception:
                    pass
                chosen = None

        if not chosen:
            # Find the first non-empty collection by checking actual stored documents
            for n in names:
                try:
                    sz = collection_size(n)
                    try:
                        print(f"[vectorstore.create_vectorstore] inspect collection '{n}' size={sz}")
                    except Exception:
                        pass
                    if sz > 0:
                        chosen = n
                        break
                except Exception:
                    try:
                        print(f"[vectorstore.create_vectorstore] error inspecting collection '{n}'")
                    except Exception:
                        pass
                    continue

        if chosen and chosen != collection_name:
            try:
                print(f"[vectorstore.create_vectorstore] default collection '{collection_name}' empty or missing; switching to non-empty collection '{chosen}'")
            except Exception:
                pass
            collection_name = chosen
    except Exception:
        # If anything goes wrong while listing collections, fall back to requested name
        pass

    # Ensure collection exists on the chromadb client side. Prefer chromadb's
    # SentenceTransformerEmbeddingFunction if available so collection is created
    # with the same embedding behavior as ingestion scripts.
    try:
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            try:
                chroma_embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            except Exception:
                chroma_embedding_fn = SentenceTransformerEmbeddingFunction()
        except Exception:
            chroma_embedding_fn = None

        if chroma_embedding_fn is not None:
            try:
                client.get_or_create_collection(name=collection_name, embedding_function=chroma_embedding_fn, metadata={"hnsw:space": "cosine"})
            except Exception:
                # Some chorma client versions may not accept embedding_function here
                try:
                    client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
                except Exception:
                    pass
        else:
            try:
                client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
            except Exception:
                pass
    except Exception:
        # If anything goes wrong, continue — Chroma wrapper may still work
        pass

    vs = Chroma(collection_name=collection_name, embedding_function=embedding_function, client=client)

    # Post-create sanity check: ensure the selected collection actually contains documents.
    try:
        coll = client.get_collection(collection_name)
        # robustly check stored data
        try:
            data = coll.get(include=['documents','metadatas','ids'])
        except Exception:
            data = coll.get(include=['documents','metadatas'])
        has_data = False
        for key in ('documents','metadatas','ids'):
            val = data.get(key)
            if val:
                if isinstance(val[0], list):
                    if len(val[0]) > 0:
                        has_data = True
                        break
                elif len(val) > 0:
                    has_data = True
                    break
        if not has_data:
            # try to find a non-empty collection and switch
            try:
                collections = client.list_collections()
                for c in collections:
                    name = c.get('name') if isinstance(c, dict) and 'name' in c else getattr(c, 'name', None)
                    if not name or name == collection_name:
                        continue
                    try:
                        ccoll = client.get_collection(name)
                        try:
                            cdata = ccoll.get(include=['documents','metadatas','ids'])
                        except Exception:
                            cdata = ccoll.get(include=['documents','metadatas'])
                        found = False
                        for key in ('documents','metadatas','ids'):
                            val = cdata.get(key)
                            if val and ((isinstance(val[0], list) and len(val[0])>0) or (len(val)>0)):
                                found = True
                                break
                        if found:
                            try:
                                print(f"[vectorstore.create_vectorstore] selected collection '{collection_name}' is empty; switching to non-empty collection '{name}' post-create")
                            except Exception:
                                pass
                            collection_name = name
                            vs = Chroma(collection_name=collection_name, embedding_function=embedding_function, client=client)
                            break
                    except Exception:
                        continue
            except Exception:
                pass
    except Exception:
        # If collection inspection fails, continue with the created vs
        pass

    try:
        print(f"[vectorstore.create_vectorstore] collection={collection_name} vectorstore={vs} embedding_fn={embedding_function}")
    except Exception:
        pass
    return vs, client


def make_multi_vector_retriever(vectorstore, docstore=None, id_key='doc_id', k=5):
    """Return a MultiVectorRetriever for the given vectorstore.

    This defers import of langchain retriever classes to avoid heavy imports
    at module import time.
    """
    # Use the resilient InMemoryStore resolved at module import time above
    docstore = docstore or InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
        search_kwargs={"k": k},
    )
    return retriever
