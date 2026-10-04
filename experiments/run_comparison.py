"""
experiments/run_comparison.py
------------------------------
Master evaluation script that executes all three architectural paradigms:
  1. Traditional ML
  2. MLOps
  3. MLSecOps (Proposed)

and runs the Data Envelopment Analysis (DEA) evaluator to produce:
  - results/benchmark_report.json
  - results/benchmark_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.traditional.run_traditional import run_traditional_experiment
from experiments.mlops.run_mlops import run_mlops_experiment
from experiments.mlsecops.run_mlsecops import run_mlsecops_experiment
from src.evaluation.dea_evaluator import DEAEvaluator
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run complete MLSecOps comparative benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Starting MLSecOps 3-Way Comparative Experiment & DEA Evaluation...")

    # 1. Run Traditional ML Baseline
    m_trad = run_traditional_experiment(config)

    # 2. Run MLOps Baseline
    m_mlops = run_mlops_experiment(config)

    # 3. Run MLSecOps Proposed Solution
    m_secops = run_mlsecops_experiment(config)

    # 4. DEA Evaluation
    evaluator = DEAEvaluator(output_dir=config["paths"]["results"])
    report = evaluator.evaluate_paradigms([m_trad, m_mlops, m_secops])

    print("\n" + "=" * 72)
    print("        MLSecOps Benchmark & DEA Evaluation Complete")
    print("=" * 72)
    print(f"  Best Paradigm: {report['best_paradigm']}  (via {report.get('ranking_method', 'dea_efficiency')})")
    print("-" * 72)
    print(f"  {'Paradigm':18s} | {'F1':>7s} | {'SecScore':>8s} | {'MTTR':>6s} | {'DEA':>6s} | {'Composite':>9s}")
    print("-" * 72)
    for p in report["paradigms"]:
        print(
            f"  {p['name']:18s} | {p['f1_score']:.4f}  | {p['security_score']:.2f}     | "
            f"{p['mttr_minutes']:5.1f}m | {p['dea_efficiency']:.4f} | {p.get('composite_score', 0):.4f}"
        )
    print("=" * 72)
    if report.get("note"):
        print(f"  Note: {report['note']}")
    print()


if __name__ == "__main__":
    main()
