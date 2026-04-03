import pandas as pd
from .arxiv_service import ArxivService
from .data_ingestion import ingest_rss_feeds

TRACKED_KEYWORDS = [
    "Federated Learning", "Edge AI", "Quantum Encryption",
    "Carbon Aware Computing", "Neuromorphic Chips", "Blockchain Identity",
    "Liquid Cooling", "Predictive Maintenance"
]

def get_rd_signals() -> pd.DataFrame:
    """
    Fetches R&D activity signals from arXiv and live RSS feeds.
    Returns a clean Market Intelligence dataset per technology keyword.
    """
    arxiv = ArxivService()
    years = [2024, 2025]

    print("Fetching arXiv R&D signals...")
    raw_data = arxiv.fetch_research_signals(TRACKED_KEYWORDS, years)

    print("Fetching live RSS signals...")
    rss_df = ingest_rss_feeds()

    rss_counts = pd.Series(dtype=int)
    if not rss_df.empty:
        exploded = rss_df.explode('keywords')
        rss_counts = exploded['keywords'].value_counts()

    records = []
    for kw in TRACKED_KEYWORDS:
        counts = raw_data.get(kw, {})
        mentions_2024 = counts.get(2024, 0)
        mentions_2025 = counts.get(2025, 0)

        live_signal = 0
        if not rss_counts.empty:
            matches = rss_counts[rss_counts.index.str.contains(kw, case=False, na=False)]
            live_signal = int(matches.sum())

        yoy_growth = 0.0
        if mentions_2024 > 0:
            yoy_growth = round(((mentions_2025 - mentions_2024) / mentions_2024) * 100, 2)

        records.append({
            "keyword": kw,
            "mentions_2024": mentions_2024,
            "mentions_2025": mentions_2025,
            "live_rss_signal": live_signal,
            "yoy_growth_pct": yoy_growth
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="mentions_2025", ascending=False)
    return df


def get_applicant_rd_breakdown(patents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyses the patent dataset to produce a company-level R&D breakdown.
    Returns top applicants ranked by filing count, broken down by tech field.
    """
    df = patents_df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Remap known column name variants
    df.rename(columns={"publication year": "year", "applicants": "applicant"}, inplace=True)

    # Generate ipc_cpc field if available
    if 'cpc classifications' in df.columns:
        df['ipc_cpc'] = df['cpc classifications'].apply(
            lambda x: str(x).split(';;')[0][:4] if pd.notna(x) and str(x) != "" else "UNKNOWN"
        )
    elif 'ipc_cpc' not in df.columns:
        df['ipc_cpc'] = "UNKNOWN"

    df.fillna("", inplace=True)

    # Filter out blank applicants
    df = df[df['applicant'].str.strip() != ""]

    if df.empty:
        return pd.DataFrame(columns=["applicant", "total_filings", "top_field", "fields_covered"])

    # Total filings per applicant
    applicant_counts = df.groupby('applicant').size().reset_index(name='total_filings')

    # Top technology field per applicant
    top_field = (
        df.groupby(['applicant', 'ipc_cpc'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .drop_duplicates(subset='applicant')
        .rename(columns={'ipc_cpc': 'top_field'})[['applicant', 'top_field']]
    )

    # Number of distinct fields covered per applicant
    fields_covered = (
        df.groupby('applicant')['ipc_cpc']
        .nunique()
        .reset_index(name='fields_covered')
    )

    # Merge all
    result = applicant_counts.merge(top_field, on='applicant', how='left')
    result = result.merge(fields_covered, on='applicant', how='left')
    result = result.sort_values(by='total_filings', ascending=False).head(15)

    return result.reset_index(drop=True)