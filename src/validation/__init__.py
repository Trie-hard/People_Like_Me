"""Statistical validation suite (PROJECT_PLAN.md §6)."""

from src.validation.coverage_evaluator import CoverageResult, ModelEvaluation, evaluate_model_loo
from src.validation.fairness_audit import FairnessReport, SubgroupCoverage, audit_age_bracket_coverage
from src.validation.winkler_score import mean_winkler, winkler_score

__all__ = [
    "winkler_score",
    "mean_winkler",
    "CoverageResult",
    "ModelEvaluation",
    "evaluate_model_loo",
    "SubgroupCoverage",
    "FairnessReport",
    "audit_age_bracket_coverage",
]
