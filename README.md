# Syllabot - A Chatbot Focused on Syllabus-Based Questions

## 📌 Overview

Syllabot is an **AI-powered academic chatbot** that allows users to upload **PDF and DOCX files** and ask questions. The chatbot uses **Milvus** as a vector database to store and retrieve document embeddings efficiently. It also integrates with **Hugging Face** for generating responses based on stored knowledge.

## 🚀 Features

- **Document Processing**: Extracts text from PDF/DOCX files.
- **Vector Storage**: Stores document embeddings in **Milvus** for efficient retrieval.
- **Querying with AI**: Uses **Sentence Transformers** for semantic search.
- **LLM Integration**: Queries **Hugging Face Hub** for generating responses.
- **User-Friendly Interface**: Built with **Streamlit** for easy interaction.

## 🛠️ Installation

### 1️⃣ Clone the Repository

```bash
git clone <repository_url>
cd syllabot-A-chatbot-focused-on-syllabus-based-questions
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables

Create a `.env` file and add the following:

```env
HUGGINGFACEHUB_API_TOKEN=<your_huggingface_api_token>
```

## ⚙️ Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

## 📜 Usage Guide

1. **Upload PDF or DOCX files** via the Streamlit interface.
2. **The chatbot processes the files** and stores embeddings in Milvus.
3. **Ask a question**, and the chatbot retrieves the most relevant document chunks.
4. **Get AI-generated answers** with references to the uploaded files.

## 🔧 Dependencies

This project uses the following libraries:

```plaintext
streamlit
pymilvus
sentence-transformers
pypdf2
langchain-community
langchain-huggingface
langchain-milvus
docx
os
```

## 📝 Milvus Setup

- Ensure you have access to **Zilliz Cloud** or a self-hosted **Milvus** instance.
- Update `app.py` with your **Milvus host, port, username, and password**.

## 🎯 Future Improvements

Based on the detailed analysis in the provided documentation, here are the key future improvements:

- **Enhanced Security Measures**:
  - Store **Milvus credentials** as environment variables instead of hardcoding them.
  - Implement **OAuth-based authentication** for secure user access.

- **Better Query Handling**:
  - Improve **duplicate detection** for document content (not just filenames).
  - Implement **Hybrid Search** (BM25 + Vector Search) to balance keyword and semantic search.
  - Cache **frequent queries** to avoid redundant computations.

- **Performance Optimization**:
  - Switch to **HNSW Indexing** for Milvus instead of IVF_FLAT for faster similarity search.
  - Dynamically adjust **nprobe** values based on query complexity to optimize retrieval.

- **Document Support Expansion**:
  - Add support for **TXT and HTML** formats in addition to PDFs and DOCX files.
  - Implement **OCR-based text extraction** for scanned PDFs.

- **Improved User Experience**:
  - Enhance the **UI with a sidebar** for document management.
  - Provide an **option to download AI-generated responses**.
  - Implement a **feedback system** for users to rate responses and improve accuracy.

- **LLM Enhancements**:
  - Integrate **better prompt engineering** to refine responses.
  - Use **multi-turn conversation support** to retain context across queries.
  - Experiment with **fine-tuned AI models** for domain-specific accuracy.

## 🤝 Contributing

Feel free to contribute! Open an issue or submit a pull request if you have improvements.



🔥 **Enjoy using Syllabot, your AI-powered academic assistant!**

