import fitz
import re
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
# from unstructured.partition.pdf import partition_pdf
import base64
import config

class PDFProcessor:
    def __init__(self, file_path):
        """
        Initialize with the PDF file path.
        """
        self.file_path = file_path
        self.image_folder = "saved_images_temp_2"
        self.pdf_document = self.read_pdf()

    def read_pdf(self):
        """
        Open and return the PDF document.
        """
        return fitz.open(self.file_path)

    def extract_metadata(self):
        """
        Extract metadata like title and author from the PDF document.
        """
        metadata = self.pdf_document.metadata
        author = metadata.get('author', 'Unknown Author')
        title = metadata.get('title', 'Untitled Document')
        return author, title

    def extract_bookmarks(self):
        """
        Extract and sort bookmarks from the PDF document.
        """
        toc = self.pdf_document.get_toc()
        bookmarks = [{'title': entry[1], 'page': entry[2]} for entry in toc]
        bookmarks.sort(key=lambda x: x['page'])  # Sort bookmarks by page number
        return bookmarks

    def enrich_and_format_pages_with_metadata(self, bookmarks, title, author):
        """
        Enrich each page of the PDF with metadata and format the text with that metadata.
        """
        formatted_texts = []
        previous_bookmark_title = None

        for page_number, page in enumerate(self.pdf_document, start=1):
            text = page.get_text("text")

            # Find the closest previous bookmark
            chapter_title = previous_bookmark_title
            for bookmark in bookmarks:
                if bookmark['page'] <= page_number:
                    chapter_title = bookmark['title']
                else:
                    break

            # Combine metadata and text into a structured format
            metadata_str = f"Title: {title}\nAuthor: {author}\nChapter: {chapter_title}\nPage Number: {page_number}\n"
            formatted_entry = {
                'metadata': metadata_str,
                'text': text
            }

            formatted_texts.append(formatted_entry)

            # Update the previous bookmark title
            for bookmark in bookmarks:
                if bookmark['page'] == page_number:
                    previous_bookmark_title = bookmark['title']

        return formatted_texts
    
    def process_pdf(self):
        """
        Process the PDF and extract elements like images and tables.
        """
        elements = partition_pdf(
            filename=self.file_path, 
            strategy="hi_res", 
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],  
            extract_image_block_to_payload=False,          
            extract_image_block_output_dir=self.image_folder
        )

    def encode_image(self, image_path):
        """
        Encode image as base64 string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def generate_img_summaries(self, image_folder):
        """
        Generate summaries and base64 encoded strings for images extracted from the PDF.
        """
        img_base64_list = []
        image_summaries = []

        if getattr(config, 'OPENAI_API_KEY', None):
            os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

        prompt = """You are an assistant tasked with summarizing images for retrieval. 
        These summaries will be embedded and used to retrieve the raw image. 
        Give a concise summary of the image that is well optimized for retrieval."""

        for img_file in sorted(os.listdir(image_folder)):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(image_folder, img_file)
                base64_image = self.encode_image(img_path)
                img_base64_list.append(base64_image)
                summary = self.image_summarize(base64_image, prompt)
                image_summaries.append({
                "image_filename": img_file,
                "summary": summary
            })
        return img_base64_list, image_summaries
    
    def image_summarize(self, img_base64, prompt):
        """Delegate image summarization to the `image_summarizer` helper."""
        from image_summarizer import summarize_image

        return summarize_image(img_base64, prompt)
    
    def extract_page_number(self, image_filename):
        """
        Extract page number from image filename, assuming the format: 'figure-10-5.jpg' or 'table-11-5.jpg'.
        The number after the first dash is considered the page number.
        """
        match = re.search(r'-(\d+)-', image_filename)
        if match:
            return int(match.group(1))
        return None
    
    def enrich_image_summaries(self, image_summaries, image_folder, author, title, bookmarks):
        """
        Enrich image summaries with metadata (author, title, page number, chapter).
        """
        enriched_image_summaries = []
        
        for image_filename in os.listdir(image_folder):
            if image_filename.endswith(".jpg"): 
                page_number = self.extract_page_number(image_filename)
                if page_number:
                    chapter_title = None
                    for bookmark in bookmarks:
                        if bookmark['page'] <= page_number:
                            chapter_title = bookmark['title']
                        else:
                            break

                    metadata = {
                        "title": title,
                        "author": author,
                        "chapter": chapter_title,
                        "page_number": page_number,
                        "image_filename": image_filename
                    }

                    for img_summary in image_summaries:
                        if img_summary.get('image_filename') == image_filename:
                            enriched_image_summaries.append({
                                "summary": img_summary['summary'],
                                "metadata": metadata
                            })
        
        return enriched_image_summaries
