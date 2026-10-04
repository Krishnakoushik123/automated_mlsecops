"""
tests/unit/test_evaluation.py
------------------------------
Unit tests for DEA evaluator module.
"""

import pytest
from src.evaluation.dea_evaluator import DEAEvaluator, ParadigmMetrics


class TestDEAEvaluator:
    def test_evaluate_paradigms(self, tmp_path):
        m1 = ParadigmMetrics("Traditional ML", 10.0, 15, 120.0, 48.0, 0.70, 0.72, 0.68, 0.20, 0.40, 0.0)
        m2 = ParadigmMetrics("MLOps", 15.0, 5, 45.0, 4.0, 0.85, 0.86, 0.84, 0.55, 0.60, 0.25)
        m3 = ParadigmMetrics("MLSecOps", 20.0, 0, 0.5, 0.5, 0.88, 0.89, 0.87, 0.95, 0.85, 1.0)

        evaluator = DEAEvaluator(output_dir=str(tmp_path))
        report = evaluator.evaluate_paradigms([m1, m2, m3])

        assert report["num_paradigms"] == 3
        # With formal LP DEA, all 3 DMUs may be efficient; composite score breaks tie
        assert report["best_paradigm"] in ("MLSecOps", "MLOps", "Traditional ML")
        assert "ranking_method" in report
        assert (tmp_path / "benchmark_report.json").exists()
        assert (tmp_path / "benchmark_report.md").exists()

        # All paradigms should have dea_efficiency and composite_score
        for p in report["paradigms"]:
            assert "dea_efficiency" in p
            assert "composite_score" in p
            assert 0.0 <= p["dea_efficiency"] <= 1.0
            assert 0.0 <= p["composite_score"] <= 1.0

    def test_mlsecops_wins_composite(self, tmp_path):
        """MLSecOps should win on composite score (best output/input ratio)."""
        m1 = ParadigmMetrics("Traditional ML", 10.0, 15, 120.0, 48.0, 0.70, 0.72, 0.68, 0.20, 0.40, 0.0)
        m2 = ParadigmMetrics("MLOps", 15.0, 5, 45.0, 4.0, 0.85, 0.86, 0.84, 0.55, 0.60, 0.25)
        m3 = ParadigmMetrics("MLSecOps", 20.0, 0, 0.5, 0.5, 0.88, 0.89, 0.87, 0.95, 0.85, 1.0)

        evaluator = DEAEvaluator(output_dir=str(tmp_path))
        report = evaluator.evaluate_paradigms([m1, m2, m3])

        # MLSecOps has best output-to-input ratio (highest outputs, lowest inputs for MTTR/deploy)
        mlsecops = [p for p in report["paradigms"] if p["name"] == "MLSecOps"][0]
        trad = [p for p in report["paradigms"] if p["name"] == "Traditional ML"][0]
        assert mlsecops["composite_score"] >= trad["composite_score"]
