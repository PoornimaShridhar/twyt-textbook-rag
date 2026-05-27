import vectorstore as vs_module


class DummyClient:
    def __init__(self, path):
        self.path = path


def test_get_client_monkeypatch(monkeypatch):
    # Replace the PersistentClient in the vectorstore module with a dummy
    monkeypatch.setattr(vs_module, "PersistentClient", lambda path: DummyClient(path))
    client = vs_module.get_client(path="my_path")
    assert isinstance(client, DummyClient)
    assert client.path == "my_path"


def test_create_vectorstore_monkeypatch(monkeypatch):
    # Monkeypatch Chroma and the embedding function to simple stand-ins
    monkeypatch.setattr(vs_module, "Chroma", lambda collection_name, embedding_function, client: {"name": collection_name})
    monkeypatch.setattr(vs_module, "SentenceTransformerEmbeddingFunction", lambda model_name=None: "emb")
    vs, client = vs_module.create_vectorstore(collection_name="col_test", client=DummyClient("p"))
    assert isinstance(vs, dict) and vs.get("name") == "col_test"
