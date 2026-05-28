# Standard library imports
import warnings
from collections import defaultdict
# Rich for enhanced console output
from rich.console import Console

# Setup rich console
console = Console(width=110)
import json
import os

class Utilities:
    def __init__(self):
        """
        Initialize the Utilities class.
        """
        pass

    def print_chunks_with_metadata(self, chunks, start=0, end=5):
        """
        Print a range of chunks along with their metadata.
        """
        for i, chunk in enumerate(chunks[start:end], start=start + 1):
            print(f"Chunk {i}:")
            print("Metadata:")
            print(chunk['metadata'])
            print("Text:")
            print(chunk['text'])
            print("-" * 40)

    def print_retrieved_results(self, query, documents, metadata):
        """
        Print the query results with documents and metadata.
        """
        print(f"\nQuery: {query}")
        print("\nRetrieved Documents and Metadata:")

        for i, (document, meta) in enumerate(zip(documents, metadata), start=1):
            print(f"Result {i}:")
            print("Document:")
            print(self.word_wrap(document))
            print("\nMetadata:")
            for key, value in meta.items():
                print(f"{key}: {value}")
            print('\n' + '-' * 40 + '\n')

    def word_wrap(self, string, n_chars=72):
        """
        Wraps a string at the next space after n_chars.
        """
        if len(string) < n_chars:
            return string
        else:
            substring = string[:n_chars]
            last_space_index = substring.rsplit(' ', 1)[0]
            wrapped_line = substring[:len(last_space_index)]
            remaining_string = string[len(wrapped_line) + 1:]
            return wrapped_line + '\n' + self.word_wrap(remaining_string, n_chars)

    def save_to_file(self, data, file_path):
        """
        Utility function to save data to a file in JSON format.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Data saved to {file_path}")


    def process_metadata(self, metadata_list):
        # Create a dictionary to store unique combinations of title and chapter
        metadata_dict = defaultdict(list)

        # Iterate through the metadata and group by title and chapter
        for entry in metadata_list:
            title = (
                entry.get('Title')
                or entry.get('title')
                or entry.get('source')
                or entry.get('file_name')
                or entry.get('filename')
                or "Unknown Source"
            )
            chapter = entry.get('Chapter') or entry.get('chapter') or "Unknown Chapter"
            page_number = (
                entry.get('Page Number')
                or entry.get('page_number')
                or entry.get('page')
                or entry.get('Page')
            )

            # Use a tuple (Title, Chapter) as the key. If page is missing or not
            # numeric, keep the source entry with an empty page list.
            key = (title, chapter)
            if page_number is None or str(page_number).strip() == "":
                metadata_dict[key]
                continue

            try:
                page_number = int(page_number)
            except (ValueError, TypeError):
                metadata_dict[key]
                continue

            metadata_dict[key].append(page_number)

        # Prepare the result in the format you specified
        result_list = []
        for (title, chapter), pages in metadata_dict.items():
            if pages:
                pages_str = ', '.join(map(str, sorted(set(pages))))
                result_list.append(f"{title}: {chapter}: {pages_str}")
            else:
                result_list.append(f"{title}: {chapter}")
        print("=========================Results of process_metadata===================================================")
        print(result_list)
        print("\n")
        return result_list
