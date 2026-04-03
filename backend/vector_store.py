import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

# ─── Persistence paths ────────────────────────────────────
STORE_DIR      = "data/faiss_store"
INDEX_PATH     = os.path.join(STORE_DIR, "index.faiss")
METADATA_PATH  = os.path.join(STORE_DIR, "metadata.pkl")
# ──────────────────────────────────────────────────────────


class PatentVectorStore:
    def __init__(self, csv_path="data/patents_clean.csv"):
        """Initialises the FAISS vector store and SentenceTransformer model."""
        self.csv_path = csv_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.df    = None
        self._load_or_build()

    # ─────────────────────────────────────────────────────────────────
    # Public helpers
    # ─────────────────────────────────────────────────────────────────

    def search_idea(self, user_idea: str, top_k: int = 5) -> dict:
        """
        Searches the vector database for patents similar to the user's idea.

        Args:
            user_idea: The technological description being validated.
            top_k:     Number of similar patents to retrieve.

        Returns:
            A dictionary containing matches, similarity scores, and risk level.
        """
        if self.index is None or self.df is None:
            return {"error": "Vector database not initialised."}

        idea_vector = self.model.encode([user_idea], convert_to_numpy=True)
        distances, indices = self.index.search(idea_vector, top_k)

        results     = []
        highest_sim = 0.0

        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            # all-MiniLM-L6-v2 produces normalised embeddings.
            # FAISS IndexFlatL2 returns d² so: CosSim ≈ 1 − (d²/2)
            sim_score = max(0.0, 1.0 - (distances[0][i] / 2.0))
            if i == 0:
                highest_sim = sim_score

            patent_info = self.df.iloc[idx].to_dict()
            patent_info['similarity'] = round(sim_score, 4)
            results.append(patent_info)

        novelty_score = round(1.0 - highest_sim, 4)

        if highest_sim > 0.80:
            risk = "HIGH"
        elif highest_sim >= 0.60:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "top_matches":        results,
            "highest_similarity": highest_sim,
            "novelty_score":      novelty_score,
            "risk_level":         risk
        }

    # ─────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────

    def _load_or_build(self):
        """
        Loads a persisted FAISS index + metadata from disk if available,
        otherwise builds from the CSV and persists for future starts.
        """
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            print("✅ [VectorStore] Saved index found — loading from disk...")
            self._load_from_disk()
        else:
            print("⚙️  [VectorStore] No saved index — building from scratch (first run only)...")
            self._build_and_save()

    def _load_from_disk(self):
        """Loads FAISS index and pre-built DataFrame from disk."""
        try:
            self.index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "rb") as f:
                self.df = pickle.load(f)
            print(f"✅ [VectorStore] Loaded {len(self.df)} patents from saved index.")
        except Exception as e:
            print(f"⚠️  [VectorStore] Failed to load saved index ({e}). Rebuilding...")
            self._build_and_save()

    def _build_and_save(self):
        """Generates embeddings, builds the FAISS index, and persists both to disk."""
        try:
            self.df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"⚠️  [VectorStore] {self.csv_path} not found. Vector store will be empty.")
            return

        # ── Normalise columns ──────────────────────────────────────
        self.df.columns = [c.lower().strip() for c in self.df.columns]
        self.df.rename(columns={"publication year": "year", "applicants": "applicant"}, inplace=True)

        if 'cpc classifications' in self.df.columns:
            self.df['ipc_cpc'] = self.df['cpc classifications'].apply(
                lambda x: str(x).split(';;')[0][:4] if pd.notna(x) and str(x) != "" else "UNKNOWN"
            )

        self.df.fillna("", inplace=True)
        self.df['search_text'] = self.df['title'] + ". " + self.df['abstract']

        # ── Generate embeddings ────────────────────────────────────
        texts      = self.df['search_text'].tolist()
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        # ── Build FAISS index ──────────────────────────────────────
        dimension  = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"✅ [VectorStore] Successfully indexed {len(texts)} patents.")

        # ── Persist to disk ────────────────────────────────────────
        os.makedirs(STORE_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.df, f)
        print(f"💾 [VectorStore] Saved index and metadata to '{STORE_DIR}/'.")
