import sys
import yaml
from pathlib import Path
from src.training.pipeline import run_pipeline

def update_config_and_run(file_path, task_type, algos):
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    cfg["dataset"]["source"] = "file"
    cfg["dataset"]["file_path"] = file_path
    cfg["dataset"]["task_type"] = task_type
    cfg["dataset"]["target_column"] = "target"
    
    cfg["training"]["algorithms"] = algos
    cfg["training"]["optimization_metric"] = "f1" if task_type == "classification" else "rmse"
    
    print(f"Running pipeline for {file_path} with {task_type}")
    try:
        summary = run_pipeline(config=cfg, experiment_id=f"test_{task_type}")
        print(f"Pipeline success for {task_type}. Best model: {summary['registry']}")
    except Exception as e:
        print(f"Pipeline failed for {task_type}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    update_config_and_run("scratch/class_data.csv", "classification", ["decision_tree", "logistic_regression"])
    update_config_and_run("scratch/reg_data.json", "regression", ["decision_tree_regressor", "linear_regression"])
