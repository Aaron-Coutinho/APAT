import plotly.graph_objects as go
import plotly.express as px

# ─── Design System ────────────────────────────────────────
BG_COLOR      = "#0B0F19"
SURFACE_COLOR = "rgba(26, 34, 53, 0.6)"
CYAN          = "#00F0FF"
PURPLE        = "#B026FF"
PINK          = "#FF2E63"
TEXT_COLOR    = "#E2E8F0"
ACCENT_GREEN  = "#00FFAB"
GOLD          = "#FFD700"

# Common layout shared across all charts
_BASE_LAYOUT = dict(
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor  = 'rgba(255,255,255,0.03)',
    font          = dict(color=TEXT_COLOR, family='Inter, sans-serif'),
    hoverlabel    = dict(
        bgcolor    = "rgba(11,15,25,0.95)",
        bordercolor= CYAN,
        font_size  = 13,
        font_family= "Inter, sans-serif"
    ),
    legend        = dict(
        font      = dict(color=TEXT_COLOR, size=12),
        bgcolor   = "rgba(0,0,0,0)",
        bordercolor= "rgba(255,255,255,0.08)",
        borderwidth= 1
    ),
)

COLORS = [CYAN, PURPLE, ACCENT_GREEN, PINK, GOLD,
          "#FF8C00", "#00BFFF", "#DA70D6", "#ADFF2F", "#FF6347"]


def _empty_chart(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(color=TEXT_COLOR, size=16, family="Inter")
    )
    fig.update_layout(**_BASE_LAYOUT, height=height)
    return fig


# ─── 1. Novelty Gauge ─────────────────────────────────────────────────────────

def create_novelty_gauge(novelty_score: float) -> go.Figure:
    val = novelty_score * 100
    if val < 25:
        bar_color = PINK
    elif val < 75:
        bar_color = GOLD
    else:
        bar_color = ACCENT_GREEN

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = val,
        delta = {'reference': 50, 'increasing': {'color': ACCENT_GREEN}, 'decreasing': {'color': PINK}},
        title = {'text': "Novelty Score (%)", 'font': {'color': TEXT_COLOR, 'size': 18, 'family': 'Inter'}},
        number= {'font': {'color': TEXT_COLOR, 'family': 'Inter', 'size': 52}, 'suffix': '%'},
        gauge = {
            'axis': {
                'range'     : [None, 100],
                'tickwidth' : 1,
                'tickcolor' : TEXT_COLOR,
                'tickfont'  : dict(color=TEXT_COLOR)
            },
            'bar'        : {'color': bar_color, 'thickness': 0.3},
            'bgcolor'    : "rgba(255, 255, 255, 0.04)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25],   'color': 'rgba(255, 46, 99, 0.12)'},
                {'range': [25, 75],  'color': 'rgba(255, 215, 0, 0.10)'},
                {'range': [75, 100], 'color': 'rgba(0, 255, 171, 0.12)'}
            ],
            'threshold': {
                'line' : {'color': "white", 'width': 3},
                'thickness': 0.85,
                'value': val
            }
        }
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        height=330,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig


# ─── 2. White-Space Quadrant ──────────────────────────────────────────────────

def create_whitespace_quadrant(df) -> go.Figure:
    df_plot = df.copy()
    df_plot['bubble_size'] = df_plot['white_space_score'].abs() + 0.5

    color_map = {
        "Goldmine (High Signal, Low Density)"  : CYAN,
        "Crowded Boom (High Signal, High Density)": PURPLE,
        "Niche/Stagnant (Low Signal, Low Density)": "#94A3B8",
        "Legacy (Low Signal, High Density)"    : PINK
    }

    fig = px.scatter(
        df_plot, x="patent_density", y="external_signal_velocity",
        size="bubble_size", color="quadrant",
        hover_name="tech_keyword",
        hover_data={
            "white_space_score"         : ":.3f",
            "bubble_size"               : False,
            "patent_density"            : ":.4f",
            "external_signal_velocity"  : ":.2f"
        },
        color_discrete_map=color_map,
        template="plotly_dark",
        labels={
            "patent_density"           : "Patent Density",
            "external_signal_velocity" : "Research Signal Velocity",
            "white_space_score"        : "WS Score"
        },
        size_max=55
    )

    avg_v = df['external_signal_velocity'].mean()
    avg_d = df['patent_density'].mean()

    fig.add_hline(y=avg_v, line_dash="dot", line_color="rgba(255,255,255,0.15)")
    fig.add_vline(x=avg_d, line_dash="dot", line_color="rgba(255,255,255,0.15)")

    # Pin quadrant labels to corners using paper coordinates — never overlaps data
    corner_labels = [
        dict(x=0.02, y=0.98, text="GOLDMINE", xanchor="left", yanchor="top",
             font=dict(size=13, color=CYAN, family="Inter", weight=700)),
        dict(x=0.98, y=0.98, text="CROWDED BOOM", xanchor="right", yanchor="top",
             font=dict(size=13, color=PURPLE, family="Inter", weight=700)),
        dict(x=0.02, y=0.02, text="NICHE", xanchor="left", yanchor="bottom",
             font=dict(size=13, color="#94A3B8", family="Inter", weight=700)),
        dict(x=0.98, y=0.02, text="LEGACY", xanchor="right", yanchor="bottom",
             font=dict(size=13, color=PINK, family="Inter", weight=700)),
    ]
    for lbl in corner_labels:
        lbl.update(xref="paper", yref="paper", showarrow=False, opacity=0.55)

    fig.update_layout(
        **_BASE_LAYOUT,
        title={'text': "White-Space Opportunity Matrix", 'font': {'size': 20, 'color': CYAN, 'family': 'Inter'}},
        xaxis=dict(tickfont=dict(color=TEXT_COLOR), title_font=dict(color=CYAN),
                   gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR), title_font=dict(color=CYAN),
                   gridcolor='rgba(255,255,255,0.06)'),
        height=520,
        annotations=corner_labels,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    return fig


# ─── 3. Technology Distribution (Treemap) ────────────────────────────────────

IPC_MAP = {
    "G06F": "Computing & Data", "G06N": "AI / ML", "H04L": "Digital Comms",
    "A61B": "MedTech", "H01L": "Semiconductors", "H04N": "Video & Signals",
    "G06Q": "FinTech", "H04W": "Wireless Comms", "G06T": "Digital Imaging",
    "H04M": "Telephony", "G01R": "Measurement", "B60W": "Autonomous Vehicles",
    "H04B": "Signal Tx", "G06K": "OCR / Recognition", "G06V": "Computer Vision"
}


def create_technology_distribution(df) -> go.Figure:
    df_plot = df.copy()
    df_plot['field_name'] = df_plot['ipc_cpc'].map(lambda x: IPC_MAP.get(x, f"Field {x}"))
    counts = df_plot['field_name'].value_counts().reset_index()
    counts.columns = ['Field', 'Count']

    fig = px.treemap(
        counts, path=['Field'], values='Count',
        color='Count', color_continuous_scale=[[0, PURPLE], [0.5, CYAN], [1, ACCENT_GREEN]],
        template="plotly_dark"
    )
    fig.update_traces(
        textfont=dict(family="Inter", size=13),
        hovertemplate="<b>%{label}</b><br>Patents: %{value:,}<extra></extra>"
    )
    fig.update_layout(**_BASE_LAYOUT, margin=dict(l=10, r=10, t=40, b=10), height=420)
    return fig


# ─── 4. Field Innovation Strength (Bar) ──────────────────────────────────────

def create_field_innovation_strength(df) -> go.Figure:
    df_plot = df.copy()
    df_plot['field_name'] = df_plot['ipc_cpc'].map(lambda x: IPC_MAP.get(x, f"Other ({x})"))
    counts = df_plot['field_name'].value_counts().reset_index()
    counts.columns = ['Field', 'Count']
    counts = counts.head(15)

    fig = px.bar(
        counts, x='Field', y='Count',
        labels={'Count': 'Number of Patents', 'Field': 'Technology Field'},
        template="plotly_dark",
        color='Count', color_continuous_scale=[[0, PURPLE], [0.5, CYAN], [1, ACCENT_GREEN]]
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Patents: %{y:,}<extra></extra>"
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(tickangle=40, tickfont=dict(color=TEXT_COLOR, size=11),
                   title_font=dict(size=13, color=CYAN), gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR), title_font=dict(size=13, color=CYAN),
                   gridcolor='rgba(255,255,255,0.05)'),
        height=430,
        margin=dict(l=20, r=20, t=50, b=110)
    )
    return fig


# ─── 5. R&D Trend Chart ───────────────────────────────────────────────────────

def create_rd_trend_chart(df) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="2024 Mentions", x=df['keyword'], y=df['mentions_2024'],
        marker=dict(color=PURPLE, opacity=0.85),
        hovertemplate="<b>%{x}</b><br>2024: %{y:,}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        name="2025 Mentions", x=df['keyword'], y=df['mentions_2025'],
        marker=dict(color=CYAN, opacity=0.95),
        hovertemplate="<b>%{x}</b><br>2025: %{y:,}<extra></extra>"
    ))

    for _, row in df.iterrows():
        sign  = "+" if row['yoy_growth_pct'] >= 0 else ""
        color = ACCENT_GREEN if row['yoy_growth_pct'] >= 0 else PINK
        fig.add_annotation(
            x=row['keyword'], y=row['mentions_2025'],
            text=f"{sign}{row['yoy_growth_pct']}%",
            showarrow=False, yshift=14,
            font=dict(size=11, color=color, family="Inter", weight=700)
        )

    fig.update_layout(
        **_BASE_LAYOUT,
        barmode='group',
        title=dict(text="R&D Activity: arXiv Mentions 2024 vs 2025",
                   font=dict(color=TEXT_COLOR, size=18, family="Inter")),
        xaxis=dict(tickangle=-30, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Mentions"),
        height=430,
        margin=dict(l=30, r=30, t=70, b=90)
    )
    return fig


# ─── 6. Applicant Landscape ───────────────────────────────────────────────────

def create_applicant_landscape_chart(df) -> go.Figure:
    if df.empty:
        return _empty_chart("No applicant data available.")

    df_s = df.sort_values('total_filings', ascending=True)

    fig = go.Figure(go.Bar(
        x=df_s['total_filings'], y=df_s['applicant'],
        orientation='h',
        marker=dict(
            color=df_s['total_filings'],
            colorscale=[[0, PURPLE], [0.5, CYAN], [1, ACCENT_GREEN]],
            showscale=False
        ),
        text=[f"{v:,}  ·  {f}" for v, f in zip(df_s['total_filings'], df_s['top_field'])],
        textposition='outside',
        textfont=dict(color=TEXT_COLOR, size=11, family="Inter"),
        hovertemplate="<b>%{y}</b><br>Filings: %{x:,}<extra></extra>"
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Top Applicants by R&D Filing Activity",
                   font=dict(color=TEXT_COLOR, size=18, family="Inter")),
        xaxis=dict(title="Total Patent Filings", gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(size=11)),
        height=max(380, len(df_s) * 40),
        margin=dict(l=20, r=150, t=60, b=30)
    )
    return fig


# ─── 7. Filing Trend Chart ────────────────────────────────────────────────────

def create_filing_trend_chart(df) -> go.Figure:
    if df.empty:
        return _empty_chart("No filing trend data available.")

    fields = df['ipc_cpc'].unique()
    fig    = go.Figure()

    for i, field in enumerate(fields):
        color    = COLORS[i % len(COLORS)]
        field_df = df[df['ipc_cpc'] == field].sort_values('year')

        fig.add_trace(go.Scatter(
            x=field_df['year'], y=field_df['filing_count'],
            mode='lines+markers', name=field,
            line=dict(color=color, width=2.5),
            marker=dict(size=7, line=dict(color='rgba(255,255,255,0.3)', width=1)),
            fill='tozeroy',
            fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.07)' if color.startswith('#') else color.replace(')', ',0.07)').replace('rgb(', 'rgba('),
            hovertemplate=f"<b>{field}</b><br>Year: %{{x}}<br>Filings: %{{y:,}}<extra></extra>"
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Patent Filing Trends by Technology Field (YoY)",
                   font=dict(color=TEXT_COLOR, size=18, family="Inter")),
        xaxis=dict(title="Year", gridcolor='rgba(255,255,255,0.05)', dtick=1),
        yaxis=dict(title="Filing Count", gridcolor='rgba(255,255,255,0.05)'),
        height=440,
    )
    return fig


# ─── 8. Problem Identification ───────────────────────────────────────────────

def create_problem_identification_chart(df) -> go.Figure:
    if df.empty:
        return _empty_chart("No problem statements found in abstracts.")

    summary = (
        df[['problem_phrase', 'frequency']]
        .drop_duplicates(subset='problem_phrase')
        .sort_values('frequency', ascending=True)
    )

    fig = go.Figure(go.Bar(
        x=summary['frequency'], y=summary['problem_phrase'],
        orientation='h',
        marker=dict(
            color=summary['frequency'],
            colorscale=[[0, PURPLE], [0.5, CYAN], [1, ACCENT_GREEN]],
            showscale=True,
            colorbar=dict(title=dict(text="Count", font=dict(color=TEXT_COLOR)),
                          tickfont=dict(color=TEXT_COLOR))
        ),
        text=summary['frequency'].apply(lambda v: f"{v:,} patents"),
        textposition='outside',
        textfont=dict(color=TEXT_COLOR, size=11, family="Inter"),
        hovertemplate="<b>%{y}</b><br>Mentions: %{x:,}<extra></extra>"
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Problem Identification: Challenges in Patent Abstracts",
                   font=dict(color=TEXT_COLOR, size=18, family="Inter")),
        xaxis=dict(title="Number of Patents", gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        height=430,
        margin=dict(l=30, r=140, t=70, b=30)
    )
    return fig


# ─── 9. Forecast Chart ────────────────────────────────────────────────────────

def create_forecast_chart(df) -> go.Figure:
    if df.empty:
        return _empty_chart("No forecast data available.")

    fields          = df['ipc_cpc'].unique()
    latest_real_yr  = int(df[df['is_forecast'] == False]['year'].max())
    fig             = go.Figure()

    for i, field in enumerate(fields):
        color    = COLORS[i % len(COLORS)]
        field_df = df[df['ipc_cpc'] == field].sort_values('year')
        hist     = field_df[field_df['is_forecast'] == False]
        fore     = field_df[field_df['is_forecast'] == True]

        if not hist.empty:
            fig.add_trace(go.Scatter(
                x=hist['year'], y=hist['filing_count'],
                mode='lines+markers', name=field,
                line=dict(color=color, width=2.5),
                marker=dict(size=7, line=dict(color='rgba(255,255,255,0.3)', width=1)),
                legendgroup=field, showlegend=True,
                hovertemplate=f"<b>{field}</b><br>Year: %{{x}}<br>Filings: %{{y:,.0f}}<extra></extra>"
            ))

        if not fore.empty and not hist.empty:
            bx = [hist['year'].iloc[-1]] + fore['year'].tolist()
            by = [hist['filing_count'].iloc[-1]] + fore['filing_count'].tolist()
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode='lines+markers',
                name=f"{field} (forecast)",
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond',
                            line=dict(color='rgba(255,255,255,0.4)', width=1)),
                legendgroup=field, showlegend=False,
                hovertemplate=f"<b>{field} ▲ Forecast</b><br>Year: %{{x}}<br>Est. Filings: %{{y:,.0f}}<extra></extra>"
            ))

    fig.add_vline(
        x=latest_real_yr + 0.5, line_dash="dot",
        line_color="rgba(255,255,255,0.35)",
        annotation_text="  TODAY ▶",
        annotation_font=dict(color="rgba(255,255,255,0.55)", size=12, family="Inter"),
        annotation_position="top left"
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Filing Forecast: Historical + 3-Year Projection",
                   font=dict(color=TEXT_COLOR, size=18, family="Inter")),
        xaxis=dict(title="Year", gridcolor='rgba(255,255,255,0.05)', dtick=1),
        yaxis=dict(title="Filing Count", gridcolor='rgba(255,255,255,0.05)'),
        height=460,
    )
    return fig


# ─── 10. Trajectory Chart ────────────────────────────────────────────────────

TRAJECTORY_COLORS = {
    "🚀 High Growth"  : CYAN,
    "📈 Steady Growth": ACCENT_GREEN,
    "➡️ Stable"       : GOLD,
    "📉 Declining"    : PINK
}


def create_trajectory_chart(df) -> go.Figure:
    if df.empty:
        return _empty_chart("No trajectory data available.")

    df_s = df.sort_values('slope', ascending=True)

    fig = go.Figure(go.Bar(
        x=df_s['slope'], y=df_s['ipc_cpc'],
        orientation='h',
        marker=dict(
            color=[TRAJECTORY_COLORS.get(t, "#94A3B8") for t in df_s['trajectory']],
            opacity=0.90,
            line=dict(color='rgba(255,255,255,0.1)', width=0.5)
        ),
        text=df_s['trajectory'],
        textposition='outside',
        textfont=dict(color=TEXT_COLOR, size=11, family="Inter"),
        hovertemplate="<b>%{y}</b><br>Slope: %{x:.2f} filings/yr<br>%{text}<extra></extra>"
    ))

    fig.add_vline(x=0, line_color="rgba(255,255,255,0.25)", line_dash="dot")

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Technology Field Trajectory Classification",
                   font=dict(color=TEXT_COLOR, size=18, family="Inter")),
        xaxis=dict(title="Growth Slope (filings/year)", gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(size=11)),
        height=max(380, len(df_s) * 42),
        margin=dict(l=30, r=200, t=70, b=30)
    )
    return fig


# ─── 11. Priority Matrix ─────────────────────────────────────────────────────

def create_priority_matrix_chart(df) -> go.Figure:
    if df.empty:
        return _empty_chart("No data available for priority matrix.")

    fig = go.Figure()

    for trajectory, color in TRAJECTORY_COLORS.items():
        subset = df[df['trajectory'] == trajectory]
        if subset.empty:
            continue

        fig.add_trace(go.Scatter(
            x=subset['total_filings'], y=subset['slope'],
            mode='markers', name=trajectory,
            text=subset['ipc_cpc'],
            marker=dict(
                size=subset['total_filings'].apply(lambda v: max(18, min(v * 6, 60))),
                color=color, opacity=0.88,
                line=dict(color='rgba(255,255,255,0.25)', width=1.5)
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Total Filings: %{x:,}<br>"
                "Growth Slope: %{y:.2f}/yr<br>"
                f"Trajectory: {trajectory}<extra></extra>"
            )
        ))

    fig.add_hline(
        y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)",
        annotation_text="Zero Growth",
        annotation_font=dict(color="rgba(255,255,255,0.4)", size=11)
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Investment Priority Matrix: Growth vs Market Presence",
                   font=dict(color=TEXT_COLOR, size=18, family="Inter")),
        xaxis=dict(title="Total Patent Filings (Market Presence)",
                   gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title="Growth Slope (filings/year)",
                   gridcolor='rgba(255,255,255,0.05)'),
        height=490,
    )
    return fig