from pathlib import Path
import pandas as pd

IN_PATH = Path("outputs/tables/category_summary.csv")
OUT_PATH = Path("outputs/tables/policy_recommendations.csv")

def action_rule(growth: float, competition: float) -> str:
    # Tunable thresholds (simple + explainable)
    if growth >= 0.15 and competition <= 0.10:
        return "High priority R&D (invest now)"
    if growth >= 0.15 and competition > 0.10:
        return "Competitive (focus on niche / differentiation)"
    if 0.05 <= growth < 0.15:
        return "Monitor (pilot projects / small investment)"
    return "Low priority (avoid for now)"

def main():
    df = pd.read_csv(IN_PATH)

    df["recommendation"] = [
        action_rule(g, c) for g, c in zip(df["growth"], df["competition_index"])
    ]

    df = df[[
        "category", "base_year", "latest_year",
        "base_value", "latest_value", "growth", "competition_index",
        "recommendation"
    ]].sort_values(by="growth", ascending=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved policy recommendations to {OUT_PATH}")

if __name__ == "__main__":
    main()