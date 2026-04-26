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

    # Build raw counts
    counts_df = (
        df.groupby(['year', 'ipc_cpc'])
        .size()
        .reset_index(name='filing_count')
    )

    # Ensure every IPC field has an entry for every year in the range
    all_years = pd.Series(range(df['year'].min(), df['year'].max() + 1), name='year')
    all_fields = counts_df['ipc_cpc'].unique()

    # Cartesian product: every year × every field
    full_grid = pd.MultiIndex.from_product(
        [all_years, all_fields],
        names=['year', 'ipc_cpc']
    ).to_frame(index=False)

    # Merge with actual counts, fill missing entries with 0
    trend_df = (
        full_grid.merge(counts_df, on=['year', 'ipc_cpc'], how='left')
        .fillna({'filing_count': 0})
        .sort_values(['ipc_cpc', 'year'])
    )

    return trend_df


def extract_problem_statements(patents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans patent abstracts for problem-indicator phrases and extracts the 
    ACTUAL problem keywords (bigrams) following the indicator.
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

    stopwords = set("the a an of in to for and or is are was were be been being have has had doing do does did but if because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t can will just don should now which that this these those from its their providing based using an".split())
    records = []

    for _, row in df.iterrows():
        abstract = str(row.get('abstract', ''))
        title = str(row.get('title', ''))
        field = str(row.get('ipc_cpc', 'UNKNOWN'))

        sentences = re.split(r'(?<=[.!?])\s+', abstract)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            for phrase in PROBLEM_INDICATORS:
                if phrase in sentence_lower:
                    # Find what comes AFTER the phrase
                    start_idx = sentence_lower.find(phrase) + len(phrase)
                    after_text = sentence_lower[start_idx:].strip()
                    
                    # Clean punctuation and find valid words
                    after_clean = re.sub(r'[^\w\s]', '', after_text)
                    words = [w for w in after_clean.split() if w not in stopwords and len(w) > 2 and not w.isdigit()]
                    
                    # Try to form a meaningful 2-word phrase describing the problem
                    actual_problem = None
                    if len(words) >= 2:
                        actual_problem = words[0] + ' ' + words[1]
                    elif len(words) == 1:
                        actual_problem = words[0]
                    else:
                        actual_problem = phrase # fallback to the indicator if empty

                    records.append({
                        "problem_phrase": actual_problem, # The specific problem like 'power consumption'
                        "indicator": phrase,             # The generic 'need for'
                        "context": sentence.strip(),
                        "ipc_cpc": field,
                        "title": title
                    })

    if not records:
        return pd.DataFrame(columns=["problem_phrase", "indicator", "context", "ipc_cpc", "title", "frequency"])

    result_df = pd.DataFrame(records)

    # Filter out fallback phrases and meaningless short phrases to get cleaner problems
    result_df = result_df[~result_df['problem_phrase'].isin(PROBLEM_INDICATORS)]
    
    if result_df.empty:
        return pd.DataFrame(columns=["problem_phrase", "indicator", "context", "ipc_cpc", "title", "frequency"])

    # Add frequency count per actual problem
    freq = result_df['problem_phrase'].value_counts().reset_index()
    freq.columns = ['problem_phrase', 'frequency']
    result_df = result_df.merge(freq, on='problem_phrase', how='left')

    # Sort by frequency descending
    result_df = result_df.sort_values('frequency', ascending=False)
    
    # We now have many specific problems. We only want to visualize the top ~20 recurring ones 
    # to avoid plotting a massive long-tail chart of 500 unique single-mentions.
    top_problems = freq.head(20)['problem_phrase'].tolist()
    result_df = result_df[result_df['problem_phrase'].isin(top_problems)]

    return result_df.reset_index(drop=True)