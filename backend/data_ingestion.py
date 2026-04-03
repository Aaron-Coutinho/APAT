import feedparser
import spacy
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import logging
import time  # ← add this

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("SpaCy model not found. Run: python -m spacy download en_core_web_sm")
    raise

RSS_SOURCES = {
    "TechCrunch":     "https://techcrunch.com/feed/",
    "MIT_Tech_Review": "https://www.technologyreview.com/feed/",
    "IEEE_Spectrum":  "https://spectrum.ieee.org/rss/fulltext"
}

# ─── RSS TTL Cache (1 hour) ────────────────────────────
_rss_cache = {"df": None, "ts": 0.0}
_RSS_CACHE_TTL = 3600  # 1 hour in seconds
# ──────────────────────────────────────────────────────

def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)

def extract_tech_keywords(text: str) -> list:
    doc = nlp(text.lower())
    keywords = []
    for chunk in doc.noun_chunks:
        if len(chunk.text) > 3 and not chunk.root.is_stop and chunk.root.pos_ != 'PRON':
            keywords.append(chunk.text.strip())
    return list(set(keywords))

def ingest_rss_feeds() -> pd.DataFrame:
    # ─── Return cached result if fresh ───
    if _rss_cache["df"] is not None and (time.time() - _rss_cache["ts"]) < _RSS_CACHE_TTL:
        logging.info("[RSS Cache HIT] Returning cached feed data.")
        return _rss_cache["df"]

    # ─── Cache miss: fetch live feeds ────
    logging.info("[RSS Cache MISS] Fetching live RSS feeds...")
    articles_data = []
    for source_name, url in RSS_SOURCES.items():
        logging.info(f"Fetching RSS feed from: {source_name}")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title        = entry.get('title', '')
                summary_html = entry.get('summary', '')
                published    = entry.get('published', datetime.now().isoformat())
                clean_summary = clean_html(summary_html)
                full_text     = f"{title}. {clean_summary}"
                keywords      = extract_tech_keywords(full_text)
                articles_data.append({
                    "source": source_name, "title": title,
                    "published_date": published, "keywords": keywords
                })
        except Exception as e:
            logging.error(f"Failed to fetch {source_name}: {e}")

    df = pd.DataFrame(articles_data) if articles_data else \
         pd.DataFrame(columns=["source", "title", "published_date", "keywords"])

    logging.info(f"Successfully ingested {len(df)} articles.")

    # ─── Store in cache ──────────────────
    _rss_cache["df"] = df
    _rss_cache["ts"] = time.time()
    return df
