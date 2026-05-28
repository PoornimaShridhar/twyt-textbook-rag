# llm-rag-conceptmap

!(home_page.png)

A multimodal Retrieval-Augmented Generation (RAG) framework for textbook-centric question answering, concept exploration, and knowledge consolidation using PDFs, YouTube content, and multimedia content.

The project integrates Chroma vector databases, LangChain pipelines, Groq-hosted LLMs, and local document processing to create an extensible educational knowledge platform.

---

## Features

- 📚 **Textbook-driven RAG pipeline**
  - Ingest PDFs and build a Chroma-backed vector store for retrieval.

- 🖼️ **Multimodal Retrieval**
  - Support for image summarization and retrieval alongside text-based content.

- 🌐 **YouTube Summarization**
  - Extend textbook knowledge with YouTube content summarization.

- 💬 **Interactive Streamlit Interface**
  - Chat-based question answering and dataset management.

- 🔄 **Incremental Knowledge Expansion**
  - Add new PDFs continuously to expand the retrieval index.

---

## Project Overview

In an era dominated by online resources, foundational textbook knowledge is often overlooked despite its importance in building deep conceptual understanding, particularly for technical interview preparation.

**Twyt Summarizer** combines Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to create an educational platform centered around textbook knowledge.

The system supports:

- Textbook-based question answering
- Concept exploration and consolidation
- YouTube content summarization
- Continuous knowledge base expansion through PDF ingestion

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Setup

Copy the example configuration file:

```bash
cp .env.example .env
```

Configure required variables:

- `GROQ_API_KEY`
- Optional `GROQ_MODEL` and `GROQ_VISION_MODEL` overrides
- Additional Chroma path or results settings as needed

Alternatively, environment variables can be set directly.

---

## Dataset Preparation

Place PDF files inside:

```text
reference_texts/
```

These documents will be processed and indexed into the vector store.

---

## Build Chroma Collection

Create embeddings and populate the Chroma database:

```bash
python chroma_collection_creator.py --chunking_strategy recursive_sentence
```

---

## Run Multimodal Retrieval

(Optional)

```bash
python multimodal_rag.py
```

---

## Launch Streamlit Interface

```bash
streamlit run chat.py
```

---

## Running Tests

```bash
pytest -q
```

---

## Models & GPU Support

### Optional Ollama Setup

Utilities under:

```text
setup_files/
```

assist with:

- Ollama installation
- GPU configuration
- Docker GPU support

Current setup targets:

- Linux environments
- NVIDIA drivers
- Docker GPU runtime

### Groq

Set:

```bash
GROQ_API_KEY=<your_key>
```

for Groq model inference.

For systems without GPU access:

- Use Groq-backed models
- Run tests using mocked LLM configurations

---

## Output Directories

Generated artifacts and retrieval outputs are stored in:

```text
results/
web_search_results/
Collections*/
```

The Chroma collections are created under the configured:

```text
CHROMA_CLIENT_PATH
```

---

## Keywords

**Retrieval-Augmented Generation (RAG)** • **Large Language Models (LLMs)** • **Technical Interview Preparation** • **In-context Learning** • **Textbook Summarization** • **Multimodal Retrieval**

---

## Tech Stack

- Python
- LangChain
- ChromaDB
- Ollama
- Groq
- Streamlit
- PDF Processing Pipelines
- Multimodal Retrieval Systems
