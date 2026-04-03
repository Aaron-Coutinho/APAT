import pandas as pd
import sys
import os

# Add the project root to sys.path to allow imports from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.arxiv_service import ArxivService
from backend.data_ingestion import ingest_rss_feeds
from backend.logic_whitespace import calculate_white_space_opportunities

def test_arxiv_service():
    print("Testing ArxivService...")
    arxiv = ArxivService()
    count = arxiv.get_mention_count_for_year("Edge AI", 2024)
    print(f"Paper count for 'Edge AI' in 2024: {count}")
    assert isinstance(count, int)
    print("ArxivService test passed!\n")

def test_rss_ingestion():
    print("Testing RSS Ingestion...")
    df = ingest_rss_feeds()
    print(f"Ingested {len(df)} articles.")
    if not df.empty:
        print("Sample keywords from first article:", df.iloc[0]['keywords'])
    assert isinstance(df, pd.DataFrame)
    print("RSS Ingestion test passed!\n")

def test_whitespace_logic():
    print("Testing Whitespace Logic with real arXiv and RSS data...")
    # Load sample patents
    try:
        patents_df = pd.read_csv("data/patents_clean.csv")
    except FileNotFoundError:
        print("Data file not found, creating mock patents...")
        patents_df = pd.DataFrame({
            "title": ["System for Edge AI", "Quantum Computing methods"],
            "abstract": ["An efficient implementation of Edge AI.", "Details about quantum computing."]
        })
    
    results = calculate_white_space_opportunities(patents_df)
    
    print("Whitespace Analysis Results (Top 3):")
    cols = ['tech_keyword', 'mentions_2024', 'mentions_2025', 'live_rss_signal', 'white_space_score', 'quadrant']
    print(results[cols].head(3))
    
    assert not results.empty
    assert 'white_space_score' in results.columns
    assert 'live_rss_signal' in results.columns
    print("Whitespace Logic test passed!")

if __name__ == "__main__":
    try:
        test_arxiv_service()
        test_rss_ingestion()
        test_whitespace_logic()
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
