import pandas as pd
import re

PROBLEM_INDICATORS = [
    "problem of", "challenge of", "need for", "limitation of",
    "difficulty in", "drawback of", "issue of", "lack of",
    "inefficiency of", "risk of", "failure of", "complexity of"
]

def get_filing_trends(patents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups patent filings by year and IPC/CPC field to produce
    year-by-year filing counts per technology sector.
    """
    df = patents_df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    df.rename(columns={"publication year": "year", "applicants": "applicant"}, inplace=True)

    if 'cpc classifications' in df.columns:
        df['ipc_cpc'] = df['cpc classifications'].apply(
            lambda x: str(x).split(';;')[0][:4] if pd.notna(x) and str(x) != "" else "UNKNOWN"
        )
    elif 'ipc_cpc' not in df.columns:
        df['ipc_cpc'] = "UNKNOWN"

    df.fillna("", inplace=True)

    # Filter out rows with no year
    df = df[df['year'].astype(str).str.strip() != ""]

    try:
        df['year'] = df['year'].astype(int)
    except Exception:
        return pd.DataFrame(columns=["year", "ipc_cpc", "filing_count"])

    trend_df = (
        df.groupby(['year', 'ipc_cpc'])
        .size()
        .reset_index(name='filing_count')
        .sort_values(['ipc_cpc', 'year'])
    )

    return trend_df


def extract_problem_statements(patents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans patent abstracts for problem-indicator phrases to identify
    what problems each technology cluster is solving.

    Returns a DataFrame with columns:
        - problem_phrase: the indicator phrase matched
        - context: the sentence containing the problem phrase
        - ipc_cpc: the technology field of the patent
        - title: the patent title
        - frequency: how often this phrase appears across all abstracts
    """
    df = patents_df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    df.rename(columns={"publication year": "year", "applicants": "applicant"}, inplace=True)

    if 'cpc classifications' in df.columns:
        df['ipc_cpc'] = df['cpc classifications'].apply(
            lambda x: str(x).split(';;')[0][:4] if pd.notna(x) and str(x) != "" else "UNKNOWN"
        )
    elif 'ipc_cpc' not in df.columns:
        df['ipc_cpc'] = "UNKNOWN"

    df.fillna("", inplace=True)

    records = []

    for _, row in df.iterrows():
        abstract = str(row.get('abstract', ''))
        title = str(row.get('title', ''))
        field = str(row.get('ipc_cpc', 'UNKNOWN'))

        # Split abstract into sentences
        sentences = re.split(r'(?<=[.!?])\s+', abstract)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            for phrase in PROBLEM_INDICATORS:
                if phrase in sentence_lower:
                    records.append({
                        "problem_phrase": phrase,
                        "context": sentence.strip(),
                        "ipc_cpc": field,
                        "title": title
                    })

    if not records:
        return pd.DataFrame(columns=["problem_phrase", "context", "ipc_cpc", "title", "frequency"])

    result_df = pd.DataFrame(records)

    # Add frequency count per phrase
    freq = result_df['problem_phrase'].value_counts().reset_index()
    freq.columns = ['problem_phrase', 'frequency']
    result_df = result_df.merge(freq, on='problem_phrase', how='left')

    # Sort by frequency descending, drop duplicates per phrase for summary view
    result_df = result_df.sort_values('frequency', ascending=False)

    return result_df.reset_index(drop=True)