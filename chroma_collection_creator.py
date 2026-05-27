import argparse
from pdf_processor import PDFProcessor
from text_processor import TextProcessor
from chroma_collection_processor import ChromaCollection
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter, CharacterTextSplitter

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and create Chroma collection based on chunking strategy.")
    parser.add_argument('--chunking_strategy', type=str, required=True, 
                        choices=['recursive', 'sentence', 'recursive_sentence', 'character'],
                        help="Chunking strategy: 'recursive', 'sentence', or 'recursive_sentence'")
    args = parser.parse_args()

    pdf_files = [
        # "elements-of-programming-interviews-java-en-316-328.pdf",
        "./reference_texts/elements-of-programming-interviews-java-en.pdf",
        "./reference_texts/Fourth_Edition_Data_Structures_and_Algor_en.pdf",
        "./reference_texts/Gayle_Laakmann-Cracking_the_Coding_Interview-en.pdf",
        "./reference_texts/The_C_Programming_Language_en.pdf",
        "./reference_texts/Data_Structures_and_Algorithms_Narasimha_Karumanchi_en.pdf"
    ]

    text_handler = TextProcessor()

    char_text_splitter = None
    token_text_splitter = None
    character_text_splitter = None
    semantic_text_splitter = None
    
    if args.chunking_strategy == 'recursive':
        char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        token_text_splitter = None
    elif args.chunking_strategy == 'sentence':
        char_text_splitter = None
        token_text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=128, chunk_overlap=10, model_name="all-MiniLM-L6-v2")
    elif args.chunking_strategy == 'recursive_sentence':
        char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        token_text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=128, chunk_overlap=10, model_name="all-MiniLM-L6-v2")
    elif args.chunking_strategy == 'character':
        character_text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=200, length_function=len, is_separator_regex=False)
    
    document_chunks = []
    metadata_chunks = []

    for pdf_file in pdf_files:
        pdf_handler = PDFProcessor(pdf_file)
        author, title = pdf_handler.extract_metadata()
        bookmarks = pdf_handler.extract_bookmarks()
        enriched_texts = pdf_handler.enrich_and_format_pages_with_metadata(bookmarks, title, author)

        if character_text_splitter:
            chunked_texts = text_handler.split_text(enriched_texts, character_text_splitter)
        elif char_text_splitter and token_text_splitter:
            chunked_texts = text_handler.split_text(enriched_texts, char_text_splitter)
            chunked_texts = text_handler.split_text(chunked_texts, token_text_splitter)
        elif char_text_splitter:
            chunked_texts = text_handler.split_text(enriched_texts, char_text_splitter)
        elif token_text_splitter:
            chunked_texts = text_handler.split_text(enriched_texts, token_text_splitter)
        elif semantic_text_splitter:
            chunked_texts = text_handler.split_text(enriched_texts, semantic_text_splitter)
        
        docs, metadata = text_handler.prepare_documents_and_metadata(chunked_texts)
        document_chunks.extend(docs)
        metadata_chunks.extend(metadata)

    collection_name = f'{args.chunking_strategy}_collection'
    
    collection_manager = ChromaCollection(collection_name)
    collection_manager.add_to_chroma_collection(document_chunks, metadata_chunks)

if __name__ == "__main__":
    main()
