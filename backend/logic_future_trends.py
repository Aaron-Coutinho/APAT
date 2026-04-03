import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def _prepare_filing_data(patents_df: pd.DataFrame) -> pd.DataFrame:
    """Internal helper: normalises and groups patents by year and field."""
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
    df = df[df['year'].astype(str).str.strip() != ""]

    try:
        df['year'] = df['year'].astype(int)
    except Exception:
        return pd.DataFrame(columns=["year", "ipc_cpc", "filing_count"])

    grouped = (
        df.groupby(['year', 'ipc_cpc'])
        .size()
        .reset_index(name='filing_count')
    )
    return grouped


def forecast_filing_trends(patents_df: pd.DataFrame, forecast_years: int = 3) -> pd.DataFrame:
    """
    Fits a linear regression per technology field and projects
    filing counts for the next N years.

    Returns combined historical + forecasted rows.
    Column 'is_forecast' = False for real data, True for projections.
    """
    grouped = _prepare_filing_data(patents_df)
    if grouped.empty:
        return pd.DataFrame()

    all_records = []
    fields = grouped['ipc_cpc'].unique()
    latest_year = int(grouped['year'].max())
    future_years = list(range(latest_year + 1, latest_year + forecast_years + 1))

    for field in fields:
        field_df = grouped[grouped['ipc_cpc'] == field].sort_values('year')

        # Add historical rows as-is
        for _, row in field_df.iterrows():
            all_records.append({
                "year": int(row['year']),
                "ipc_cpc": field,
                "filing_count": int(row['filing_count']),
                "is_forecast": False
            })

        # Need at least 2 data points to fit a line
        if len(field_df) < 2:
            continue

        X = field_df['year'].values.reshape(-1, 1)
        y = field_df['filing_count'].values

        model = LinearRegression()
        model.fit(X, y)

        for fy in future_years:
            predicted = model.predict([[fy]])[0]
            # Clamp to 0 — filing counts can't be negative
            predicted = max(0, round(predicted, 2))
            all_records.append({
                "year": fy,
                "ipc_cpc": field,
                "filing_count": predicted,
                "is_forecast": True
            })

    result = pd.DataFrame(all_records).sort_values(['ipc_cpc', 'year'])
    return result.reset_index(drop=True)


def classify_trajectory(patents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies each technology field by its growth trajectory
    using the slope of its linear regression line.

    Trajectory labels:
        🚀 High Growth   — slope > 1.5
        📈 Steady Growth — slope between 0.2 and 1.5
        ➡️ Stable        — slope between -0.2 and 0.2
        📉 Declining     — slope < -0.2
    """
    grouped = _prepare_filing_data(patents_df)
    if grouped.empty:
        return pd.DataFrame()

    records = []
    fields = grouped['ipc_cpc'].unique()

    for field in fields:
        field_df = grouped[grouped['ipc_cpc'] == field].sort_values('year')

        total_filings = int(field_df['filing_count'].sum())

        if len(field_df) < 2:
            slope = 0.0
        else:
            X = field_df['year'].values.reshape(-1, 1)
            y = field_df['filing_count'].values
            model = LinearRegression()
            model.fit(X, y)
            slope = round(model.coef_[0], 4)

        if slope > 1.5:
            trajectory = "🚀 High Growth"
            priority = "HIGH"
        elif slope > 0.2:
            trajectory = "📈 Steady Growth"
            priority = "MEDIUM"
        elif slope >= -0.2:
            trajectory = "➡️ Stable"
            priority = "LOW"
        else:
            trajectory = "📉 Declining"
            priority = "LOW"

        records.append({
            "ipc_cpc": field,
            "slope": slope,
            "trajectory": trajectory,
            "priority": priority,
            "total_filings": total_filings
        })

    result = pd.DataFrame(records).sort_values('slope', ascending=False)
    return result.reset_index(drop=True)


def generate_rd_recommendations(patents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines trajectory classification to produce ranked,
    actionable R&D policy recommendations per technology field.

    Returns a DataFrame with:
        - priority: HIGH / MEDIUM / LOW
        - field: IPC/CPC code
        - trajectory: growth label
        - recommendation: plain-English action text
        - reason: data-backed justification
    """
    trajectory_df = classify_trajectory(patents_df)
    if trajectory_df.empty:
        return pd.DataFrame()

    records = []

    for _, row in trajectory_df.iterrows():
        field = row['ipc_cpc']
        trajectory = row['trajectory']
        slope = row['slope']
        priority = row['priority']
        total = row['total_filings']

        if "High Growth" in trajectory:
            recommendation = f"Aggressively invest R&D resources in field {field}."
            reason = (
                f"Filing activity is growing steeply (slope: +{slope}/yr) "
                f"with {total} total filings. Early R&D investment will secure "
                f"a strong IP position before the field becomes crowded."
            )
        elif "Steady Growth" in trajectory:
            recommendation = f"Maintain and gradually increase R&D spend in field {field}."
            reason = (
                f"Field {field} shows consistent positive growth (slope: +{slope}/yr) "
                f"across {total} filings. Sustaining investment will protect "
                f"existing IP while capturing incremental opportunities."
            )
        elif "Stable" in trajectory:
            recommendation = f"Monitor field {field} — hold current R&D investment."
            reason = (
                f"No significant growth or decline detected (slope: {slope}/yr) "
                f"across {total} filings. Reallocate freed budget toward "
                f"High Growth fields unless strategic value exists."
            )
        else:
            recommendation = f"Reduce R&D exposure in field {field} or pivot to adjacent areas."
            reason = (
                f"Filing activity is declining (slope: {slope}/yr) across "
                f"{total} total filings. Continued heavy investment risks diminishing "
                f"returns — consider licensing existing IP instead."
            )

        records.append({
            "priority": priority,
            "field": field,
            "trajectory": trajectory,
            "recommendation": recommendation,
            "reason": reason
        })

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    result = pd.DataFrame(records)
    result['sort_key'] = result['priority'].map(priority_order)
    result = result.sort_values('sort_key').drop(columns='sort_key')

    return result.reset_index(drop=True)