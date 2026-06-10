"""Tag every bank item with family + difficulty-axis metadata (Week 3).

Emits a tidy table — one row per item — whose columns are exactly the predictors
the Week 7 difficulty regression needs: estimated difficulty b ~ family +
tightness + n_objectives + corner. Saved as item_metadata.{parquet,csv}.
"""
from __future__ import annotations

import json
import math

import pandas as pd

from circuit_irt.paths import DATA

# ordinal "robustness/corner" severity for the regression (none = 0)
CORNER_SEVERITY = {
    "none": 0,
    "comp_tol_10pct": 1, "supply_pm10": 1, "cload_1_10pf": 1,
    "comp_tol_5pct": 2, "temp_0_85C": 2, "tail_finite_ro": 2,
    "vth_corner": 3, "mismatch_1pct": 3,
}


def build_metadata(bank_path=DATA / "candidate_bank.json") -> pd.DataFrame:
    items = json.load(open(bank_path))["items"]
    rows = []
    for it in items:
        corner = it["corner"]
        rows.append({
            "item_id": it["item_id"],
            "family": it["family_id"],
            # --- difficulty axis 1: constraint tightness ---
            "tightness": it["tightness"],
            "log2_tightness": math.log2(it["tightness"]),
            # --- difficulty axis 2: # simultaneous objectives ---
            "n_objectives": it["n_objectives"],
            # --- difficulty axis 3: robustness / corner ---
            "corner": corner,
            "has_corner": corner != "none",
            "corner_severity": CORNER_SEVERITY.get(corner, 1),
            # composite + bookkeeping
            "tier": it["tier"],
            "difficulty_score": it["difficulty_score"],
            "objectives": ",".join(it["objectives"]),
            "reliability_subset": it.get("reliability_subset", False),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_metadata()
    df.to_parquet(DATA / "item_metadata.parquet", index=False)
    df.to_csv(DATA / "item_metadata.csv", index=False)

    print(f"tagged {len(df)} items -> item_metadata.{{parquet,csv}}\n")
    print("columns:", list(df.columns), "\n")
    print("by family:\n", df["family"].value_counts().to_string())
    print("\ntightness x n_objectives (counts):")
    print(pd.crosstab(df["tightness"], df["n_objectives"]).to_string())
    print("\ncorner distribution:\n", df["corner"].value_counts().to_string())
    # sanity: every difficulty axis actually varies (regression needs variance)
    for col in ("tightness", "n_objectives", "corner_severity"):
        assert df[col].nunique() > 1, f"{col} has no variance"
    assert df["item_id"].is_unique and len(df) >= 200
    print("\nOK: difficulty-axis metadata tagged with variance on every axis.")
