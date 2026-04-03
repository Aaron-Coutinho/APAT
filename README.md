# Patent Intelligence Platform 🧬

AI-powered IP analytics suite designed for semantic patent validation and white-space discovery. The platform utilizes FAISS for high-performance vector search and integrates real-time research signals from arXiv and RSS feeds to identify innovation opportunities.

## 🚀 Key Modules

- **💡 Idea Validator**: Assess the novelty of your ideas against millions of patents using semantic embeddings (`all-MiniLM-L6-v2`).
- **🚀 White-Space Discovery**: Identify market gaps by cross-referencing high-velocity research signals with existing patent density.
- **📊 Patent Explorer**: Dive deep into technological neighbors using advanced semantic clustering.
- **🌐 IP Landscape**: Visualize global innovation strength and sector distribution with interactive dynamic charts.

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/) with custom Plotly components.
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Asynchronous API).
- **Vector Engine**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search).
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`).
- **External Signals**: arXiv API Integration & Multi-source RSS Ingestion.

## 📦 Installation & Setup

### 1. Prerequisite
 Python 3.11+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Data Preparation
Place your cleaned patent database in the data directory:
`data/patents_clean.csv`

## 🏃 Running the Application

For a fully functional experience, you must run both the backend services and the frontend dashboard.

#### Step 1: Start the Backend (FastAPI)
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
*Wait for the "Successfully indexed patents" message before proceeding.*

#### Step 2: Start the Frontend (Streamlit)
```bash
python -m streamlit run frontend/app.py --server.port 8501
```

## 🧪 Verification & Testing
To ensure all external services (arXiv/RSS) and data mappings are functional, run the integration suite:
```bash
python tests/verify_arxiv_integration.py
```


