from pdf_processor import PDFProcessor
from text_processor import TextProcessor
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import os
import uuid
from langchain_core.documents import Document
from vectorstore import InMemoryStore, MultiVectorRetriever
import config
import pickle

def parse_metadata(metadata_str):
    """
    Convert a metadata string into a dictionary by splitting it by lines and colons.
    Example:
    'Title: Elements of Programming Interviews in Java\nAuthor: Adnan Aziz\n...'
    becomes
    {'Title': 'Elements of Programming Interviews in Java', 'Author': 'Adnan Aziz', ...}
    """
    metadata_dict = {}
    for line in metadata_str.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            metadata_dict[key.strip()] = value.strip()
    return metadata_dict


def create_multi_vector_retriever(vectorstore, text_summaries, texts, image_summaries):
    """
    Create retriever that indexes summaries but returns raw images, tables, or texts along with metadata.
    """
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs={"k": 5}
    )
    def add_documents(retriever, doc_summaries):
        doc_ids = [str(uuid.uuid4()) for _ in doc_summaries]
        summary_docs = []
        
        for i, chunk in enumerate(doc_summaries):
            text = chunk['text']
            metadata_str = chunk['metadata']
            if isinstance(metadata_str, str):
                metadata = parse_metadata(metadata_str)
            else:
                metadata = metadata_str
            doc = Document(page_content=text, metadata={id_key: doc_ids[i], **metadata})
            summary_docs.append(doc)
            
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, [chunk['text'] for chunk in doc_summaries])))  # Store the raw content with ID

    if text_summaries:
        add_documents(retriever, text_summaries)  # You can adjust table metadata

    if image_summaries:
        image_summary_texts = [img_summary['summary'] for img_summary in image_summaries]
        image_metadata = [img_summary['metadata'] for img_summary in image_summaries]
        add_documents(retriever, [{'text': img_summary, 'metadata': img_metadata} for img_summary, img_metadata in zip(image_summary_texts, image_metadata)])

    return retriever


def main():
    pdf_files = [
        # "elements-of-programming-interviews-java-en-316-328.pdf",
        "./reference_texts/elements-of-programming-interviews-java-en.pdf",
        "./reference_texts/Fourth_Edition_Data_Structures_and_Algor_en.pdf",
        "./reference_texts/Gayle_Laakmann-Cracking_the_Coding_Interview-en.pdf",
        "./reference_texts/The_C_Programming_Language_en.pdf",
        "./reference_texts/Data_Structures_and_Algorithms_Narasimha_Karumanchi_en.pdf"
    ]

    text_handler = TextProcessor()
    char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    token_text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=128, chunk_overlap=10, model_name="all-MiniLM-L6-v2")
    document_chunks = []
    image_summaries = []

    for pdf_file in pdf_files:
        pdf_handler = PDFProcessor(pdf_file)
        author, title = pdf_handler.extract_metadata()
        bookmarks = pdf_handler.extract_bookmarks()
        enriched_texts = pdf_handler.enrich_and_format_pages_with_metadata(bookmarks, title, author)
        print("Author : ", author)
        print("Title : ", title)
        print("--------------------------------------------------------------------------")

        chunked_texts = text_handler.split_text(enriched_texts, char_text_splitter)
        chunked_texts = text_handler.split_text(chunked_texts, token_text_splitter)
        
        document_chunks.extend(chunked_texts)
    
    collection_name = config.DEFAULT_COLLECTION_NAME or "text_Collection_3"
    import vectorstore
    vs, client = vectorstore.create_vectorstore(collection_name=collection_name)

    retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore=vs,
    text_summaries=document_chunks, 
    texts=document_chunks,  
    image_summaries=image_summaries,  
    )
    retriever_multi_vector_img.invoke("What are the types of databases?")

if __name__ == "__main__":
    main()
