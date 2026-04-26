import pandas as pd
import json
import os
import logging
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── Disk cache paths ──────────────────────────────────────
TOPIC_CACHE_DIR  = "data/topic_cache"
SUMMARY_PATH     = os.path.join(TOPIC_CACHE_DIR, "topic_summary.json")
PATENTS_PATH     = os.path.join(TOPIC_CACHE_DIR, "patents_with_topics.json")
# ──────────────────────────────────────────────────────────


class PatentTopicModeler:
    """
    Discovers technology clusters in patent data using BERTopic.

    On the first call (or when force=True), topics are computed and
    saved to disk for instant loading on subsequent server restarts.
    Manual re-computation: pass force=True or delete the cache directory.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.topic_model     = BERTopic(embedding_model=self.embedding_model, verbose=True)
        self.topics          = None
        self.probs           = None

    # ─── Public interface ──────────────────────────────────

    def get_topics_for_api(self, patents_df: pd.DataFrame, force: bool = False) -> dict:
        """
        Returns topic summary and patent-to-topic mapping.

        Loads from disk cache if available (and force=False),
        otherwise runs full BERTopic and saves results.

        Args:
            patents_df: The full patents DataFrame from VectorStore.
            force:      If True, ignores cache and re-runs BERTopic.

        Returns:
            dict with keys 'topic_summary' and 'patents_with_topics'.
        """
        if not force and self._cache_exists():
            logging.info("[TopicModeler] Loading topics from disk cache...")
            return self._load_from_disk()

        logging.info("[TopicModeler] Cache miss or force=True — running BERTopic...")
        return self._run_and_save(patents_df)

    # ─── BERTopic core ────────────────────────────────────

    def fit_transform(self, documents: List[str]) -> Tuple[List[int], List[float]]:
        logging.info(f"Starting topic modeling on {len(documents)} documents...")
        self.topics, self.probs = self.topic_model.fit_transform(documents)
        logging.info("Topic modeling complete.")
        return self.topics, self.probs

    def get_topic_info(self) -> pd.DataFrame:
        if self.topic_model is None:
            return pd.DataFrame()
        return self.topic_model.get_topic_info()

    def extract_topics_from_patents(self, patents_df: pd.DataFrame) -> pd.DataFrame:
        patents_df.columns = [c.lower().strip() for c in patents_df.columns]
        patents_df['full_text'] = patents_df['title'] + ". " + patents_df['abstract']
        docs = patents_df['full_text'].tolist()

        topics, _ = self.fit_transform(docs)

        topic_info    = self.get_topic_info()
        topic_mapping = dict(zip(topic_info['Topic'], topic_info['Name']))

        patents_df['topic_id']    = topics
        patents_df['topic_label'] = [topic_mapping.get(t, "Unknown") for t in topics]

        return patents_df

    # ─── Disk cache helpers ────────────────────────────────

    def _cache_exists(self) -> bool:
        return os.path.exists(SUMMARY_PATH) and os.path.exists(PATENTS_PATH)

    def _load_from_disk(self) -> dict:
        try:
            with open(SUMMARY_PATH, "r") as f:
                topic_summary = json.load(f)
            with open(PATENTS_PATH, "r") as f:
                patents_with_topics = json.load(f)
            logging.info(f"[TopicModeler] Loaded {len(topic_summary)} topics from cache.")
            return {"topic_summary": topic_summary, "patents_with_topics": patents_with_topics}
        except Exception as e:
            logging.warning(f"[TopicModeler] Failed to read cache ({e}). Re-running BERTopic.")
            return {}

    def _run_and_save(self, patents_df: pd.DataFrame) -> dict:
        try:
            result_df  = self.extract_topics_from_patents(patents_df.copy())
            topic_info = self.get_topic_info()

            topic_summary       = topic_info.to_dict(orient="records")
            patents_with_topics = result_df[['title', 'topic_id', 'topic_label']].to_dict(orient="records")

            os.makedirs(TOPIC_CACHE_DIR, exist_ok=True)
            with open(SUMMARY_PATH, "w") as f:
                json.dump(topic_summary, f)
            with open(PATENTS_PATH, "w") as f:
                json.dump(patents_with_topics, f)

            logging.info(f"[TopicModeler] Saved topic cache to '{TOPIC_CACHE_DIR}/'.")
            return {"topic_summary": topic_summary, "patents_with_topics": patents_with_topics}
        except Exception as e:
            logging.error(f"[TopicModeler] BERTopic failed: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    """
    Manual precompute / force-rebuild entry point.
    Usage:
        python -m backend.topic_modeling            # build if no cache
        python -m backend.topic_modeling --force    # always rebuild
    """
    import sys
    force = "--force" in sys.argv

    print(f"Running topic modeling (force={force})...")
    df = pd.read_csv("data/patents_clean.csv")
    modeler = PatentTopicModeler()
    result  = modeler.get_topics_for_api(df, force=force)

    if "error" in result:
        print(f"❌ Failed: {result['error']}")
    else:
        print(f"✅ Done. {len(result['topic_summary'])} topics saved to '{TOPIC_CACHE_DIR}/'.")
