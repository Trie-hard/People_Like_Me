"""CLI: LOO coverage, Winkler comparison, fairness audit (subset for runtime)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.validation import audit_age_bracket_coverage, evaluate_model_loo


def main() -> None:
    p = argparse.ArgumentParser(description="Run validation battery on pooled MEC cohort")
    p.add_argument("--parquet", type=Path, default=Path("data/processed/mec_cohort_pooled.parquet"))
    p.add_argument("--analyte", default="lbxgh")
    p.add_argument("--max-units", type=int, default=300, help="LOO subsample size per model")
    p.add_argument("--k", type=int, default=300)
    args = p.parse_args()
    if not args.parquet.exists():
        raise SystemExit(f"Missing {args.parquet}; run scripts/build_nhanes_parquet.py first.")

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df):,} MEC rows from {args.parquet}\n")

    for model in ("baseline0", "baseline1", "knn"):
        ev = evaluate_model_loo(
            df,
            analyte=args.analyte,
            model=model,
            k=args.k,
            max_units=args.max_units,
        )
        print(f"=== {ev.model_name} ===")
        print(f"Mean Winkler @ 90%: {ev.mean_winkler_90:.4f}")
        for c in ev.coverages:
            print(
                f"  alpha={c.alpha:.2f}  weighted_cov={c.weighted_coverage:.3f}  "
                f"unweighted={c.unweighted_coverage:.3f}  n={c.n_evaluated}"
            )
        print()

    fair = audit_age_bracket_coverage(df, analyte=args.analyte, max_units_per_group=min(120, args.max_units))
    print("=== Fairness (age brackets, 90% k-NN coverage) ===")
    print(fair.notes)
    for g in fair.groups:
        if g.suppressed:
            print(f"  {g.label}: suppressed (eff_n={g.effective_n:.0f}, n={g.n_units})")
        else:
            print(f"  {g.label}: coverage={g.weighted_coverage_90:.3f} (n={g.n_units})")
    if fair.flagged:
        print("  ** Humility note: coverage gap exceeds policy threshold **")


if __name__ == "__main__":
    main()
