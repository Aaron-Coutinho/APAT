import pandas as pd
import numpy as np
from .arxiv_service import ArxivService
from .data_ingestion import ingest_rss_feeds

def fetch_arxiv_signals(keywords: list = None) -> pd.DataFrame:
    """
    Fetches real research activity data from arXiv for 2024 and 2025.
    
    Returns:
        pd.DataFrame: A dataset with mentions per technology.
    """
    if keywords is None:
        keywords = [
            "Federated Learning", "Edge AI", "Quantum Encryption", 
            "Carbon Aware Computing", "Neuromorphic Chips", "Blockchain Identity",
            "Liquid Cooling", "Predictive Maintenance"
        ]
    
    arxiv = ArxivService()
    years = [2024, 2025]
    
    print(f"Fetching arXiv signals for {len(keywords)} keywords...")
    raw_arxiv_data = arxiv.fetch_research_signals(keywords, years)
    
    # Also fetch live RSS data
    print("Fetching live RSS signals...")
    rss_df = ingest_rss_feeds()
    
    rss_counts = pd.Series()
    if not rss_df.empty:
        # Explode keywords and count occurrences
        exploded_keywords = rss_df.explode('keywords')
        rss_counts = exploded_keywords['keywords'].value_counts()
    
    processed_records = []
    for kw in keywords:
        # Combine arXiv velocity and activity with RSS frequency
        # arXiv data
        counts = raw_arxiv_data.get(kw, {})
        mentions_2024 = counts.get(2024, 0)
        mentions_2025 = counts.get(2025, 0)
        
        # RSS data (live signal)
        # We match keyword case-insensitively or exactly as per SpaCy output
        # Let's try to match within the index
        live_signal = 0
        if not rss_counts.empty:
            matches = rss_counts[rss_counts.index.str.contains(kw, case=False, na=False)]
            live_signal = matches.sum()
            
        processed_records.append({
            "tech_keyword": kw,
            "mentions_2024": mentions_2024,
            "mentions_2025": mentions_2025,
            "live_rss_signal": live_signal
        })
        
    return pd.DataFrame(processed_records)

def calculate_white_space_opportunities(patents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies high-growth 'White Space' using volume-weighted opportunity scores.
    
    Statistical Formulation:
    - Velocity = (M_2025 - M_2024) / M_2024
    - ResearchActivity = M_2025 + (Live_RSS_Signal * Weight)
    - Normalized PatentDensity = Patents matching Keyword / Total Patents
    - WhiteSpaceScore = (Velocity * ResearchActivity) / (PatentDensity + 0.001)

    Args:
        patents_df: The local dataset of processed patents.

    Returns:
        pd.DataFrame: Opportunities ranked by White-Space Score.
    """
    signals_df = fetch_arxiv_signals()
    total_patents = len(patents_df)
    
    # Normalize column names to lowercase for robust indexing
    patents_df.columns = [c.lower() for c in patents_df.columns]
    
    # Calculate Statistical Components
    signals_df['external_signal_velocity'] = (signals_df['mentions_2025'] - signals_df['mentions_2024']) / (signals_df['mentions_2024'] + 0.001)
    
    # Combined Research Activity: arXiv 2025 + Live RSS Weight (arbitrary weight of 5 for RSS mentions)
    signals_df['research_activity'] = signals_df['mentions_2025'] + (signals_df['live_rss_signal'] * 5)
    
    # Calculate Normalized Patent Density per keyword
    density_records = []
    for keyword in signals_df['tech_keyword']:
        # Keywords matched against title and abstract
        count = patents_df['title'].str.contains(keyword, case=False, na=False).sum() + \
                patents_df['abstract'].str.contains(keyword, case=False, na=False).sum()
        
        # Calculate patent density as a fraction of the entire database
        normalized_density = count / total_patents if total_patents > 0 else 0
        density_records.append({"tech_keyword": keyword, "patent_density": normalized_density})
        
    density_df = pd.DataFrame(density_records)
    
    # Merge and Calculate Score
    merged_df = pd.merge(signals_df, density_df, on="tech_keyword")
    
    # Implement Volume-Weighted Algorithm
    # Using epsilon (0.001) to ensure continuity and prevent division by zero
    merged_df['white_space_score'] = (merged_df['external_signal_velocity'] * merged_df['research_activity']) / (merged_df['patent_density'] + 0.001)
    
    # Sector Classification (Quadrant Analysis)
    # We use the dataset's current mean metrics to divide the quadrants
    avg_velocity = merged_df['external_signal_velocity'].mean()
    avg_density = merged_df['patent_density'].mean()
    
    def assign_quadrant(row):
        if row['external_signal_velocity'] > avg_velocity and row['patent_density'] < avg_density:
            return "Goldmine (High Signal, Low Density)"
        elif row['external_signal_velocity'] > avg_velocity and row['patent_density'] >= avg_density:
            return "Crowded Boom (High Signal, High Density)"
        elif row['external_signal_velocity'] <= avg_velocity and row['patent_density'] < avg_density:
            return "Niche/Stagnant (Low Signal, Low Density)"
        else:
            return "Legacy (Low Signal, High Density)"
            
    merged_df['quadrant'] = merged_df.apply(assign_quadrant, axis=1)
    
    # Sort by White-Space Score for prioritized strategic focus
    return merged_df.sort_values(by="white_space_score", ascending=False)
