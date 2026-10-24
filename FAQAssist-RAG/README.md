# FAQAssist-RAG
CSV-Based FAQ Question Answering using Vector Search and LLMs

FAQAssist-RAG is an application-level Retrieval-Augmented Generation (RAG) system designed for online course institutes. It answers user questions by retrieving relevant information from a structured FAQ dataset stored in a CSV file and generating responses strictly grounded in that data using a large language model.

The project demonstrates how structured data (FAQs) can be transformed into a semantic search system using embeddings, a vector database, and an LLM.

---

## 🚀 Key Features
- Uses a CSV file as the knowledge base (FAQ dataset)
- Converts FAQ data into vector embeddings
- Stores embeddings in a FAISS vector database
- Retrieves relevant FAQ entries using semantic search
- Generates answers using a Gemini LLM
- Ensures responses are grounded only in the provided data
- Interactive web interface built with Streamlit

---

## 🧠 How the System Works (RAG Pipeline)

1. **Data Loading**
   - FAQ data is loaded from a CSV file using `CSVLoader`
   - Each FAQ entry is treated as a document

2. **Embedding**
   - Documents are converted into vector embeddings using:
     - `sentence-transformers/all-MiniLM-L6-v2`

3. **Vector Storage**
   - Embeddings are stored locally using FAISS
   - The vector index is generated once and reused

4. **Retrieval**
   - When a user asks a question, semantically similar FAQ entries are retrieved

5. **Generation**
   - Retrieved context is passed to a Gemini-based LLM
   - A strict prompt ensures answers are based only on the retrieved context
   - If no relevant data is found, the system responds with *"I don't know."*

---

## 🧩 Tech Stack

- **Frontend / UI**: Streamlit
- **LLM**: Google Gemini (`gemini-2.0-flash`)
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Database**: FAISS
- **Framework**: LangChain
- **Language**: Python

---

## 📁 Project Structure

```text
FAQAssist-RAG/
├── app/
│   ├── main.py              # Streamlit application
│   └── langchain_utils.py   # Vector DB creation and RAG pipeline
│
├── data/
│   └── codebasics_faqs.csv  # Demo FAQ dataset
│
├── notebooks/
│   └── Notebook.ipynb       # Experimentation and testing
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
