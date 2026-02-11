import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Load existing trend table
TREND_PATH = Path("outputs/tables/trend_table.csv")

st.title("Automated Patentability Analytics Demo")

st.header("Enter Your Idea")

title = st.text_input("Patent Title")
abstract = st.text_area("Patent Abstract")

if st.button("Analyze Idea"):

    if not abstract:
        st.warning("Please enter abstract")
    else:
        # Load trend data
        trend_df = pd.read_csv(TREND_PATH, index_col=0)

        # Extract keywords from user abstract
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        vectorizer.fit([abstract])
        keywords = vectorizer.get_feature_names_out()

        st.subheader("Extracted Keywords from Idea")
        st.write(keywords)

        # Match keywords with trend table
        matched = [k for k in keywords if k in trend_df.index]

        if matched:
            st.subheader("Matching Market Trends")
            st.write(matched)

            latest_year = trend_df.columns[-1]

            for k in matched:
                score = trend_df.loc[k, latest_year]
                st.write(f"{k} → Market Score ({latest_year}): {round(score,4)}")
        else:
            st.info("No direct keyword match found in current dataset.")

        st.subheader("Current Market Trend Overview")
        st.image("outputs/figures/trend_top_keywords_by_year.png")

        st.subheader("White Space Candidates")
        ws_path = Path("outputs/tables/white_space_candidates.csv")
        if ws_path.exists():
            ws_df = pd.read_csv(ws_path)
            st.dataframe(ws_df.head(10))
        else:
            st.info("White space candidate data not available yet.")

