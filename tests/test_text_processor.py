from text_processor import TextProcessor


class DummySplitter:
    def split_text(self, text):
        # Return two chunks for any input
        return ["chunk1", "chunk2"]


def test_split_and_prepare_documents_and_metadata():
    tp = TextProcessor()
    formatted_texts = [{"metadata": "Title: TestTitle\nAuthor: TestAuthor\n", "text": "Some long text"}]

    chunks = tp.split_text(formatted_texts, DummySplitter())
    assert isinstance(chunks, list)
    assert len(chunks) == 2

    docs, metadata = tp.prepare_documents_and_metadata(chunks)
    assert docs == ["chunk1", "chunk2"]
    assert isinstance(metadata, list)
    # metadata should include parsed Title
    assert metadata[0].get("Title") == "TestTitle"
