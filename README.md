# PDF Q&A Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload PDF documents and ask questions in natural language. The system retrieves relevant document chunks using semantic search and generates context-aware answers using a Large Language Model (LLM).

## Features

* Upload and process PDF documents
* AI-powered Question Answering
* PDF Summary Generation
* Source Page Tracking
* Retrieved Chunk Viewer
* FAISS Vector Search
* Groq LLaMA 3.1 Integration

## Tech Stack

* Streamlit
* LangChain
* FAISS
* HuggingFace MiniLM Embeddings
* Groq (LLaMA 3.1)

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Project Structure

```text
pdf-qa-chatbot/
├── app.py
├── requirements.txt
├── README.md
```

## Future Improvements

* Multi-PDF Support
* Chat History Export
* OCR Support for Scanned PDFs
* Enhanced Citation Highlighting


