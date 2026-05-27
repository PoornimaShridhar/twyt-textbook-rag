# llm-rag-conceptmap

A multimodal Retrieval-Augmented Generation (RAG) framework for textbook-centric question answering, concept exploration, and knowledge consolidation using PDFs, web resources, and multimedia content.

The project integrates Chroma vector databases, LangChain pipelines, LLMs (OpenAI / Ollama), and local document processing to create an extensible educational knowledge platform.

---

## Features

- 📚 **Textbook-driven RAG pipeline**
  - Ingest PDFs and build a Chroma-backed vector store for retrieval.

- 🖼️ **Multimodal Retrieval**
  - Support for image summarization and retrieval alongside text-based content.

- 🌐 **Web & YouTube Summarization**
  - Extend textbook knowledge with web search and YouTube content summarization.

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
- Web search summarization
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

- `OPENAI_API_KEY`
- `OLLAMA_URL`
- Additional model or embedding configurations as needed

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

### Ollama

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

### OpenAI

Set:

```bash
OPENAI_API_KEY=<your_key>
```

for OpenAI embeddings and model inference.

For systems without GPU access:

- Use OpenAI-backed models
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
- OpenAI
- Streamlit
- PDF Processing Pipelines
- Multimodal Retrieval Systems
