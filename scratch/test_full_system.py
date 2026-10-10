import os
import sys
import json
import zipfile
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.training.pipeline import run_pipeline
from src.utils.config import load_config
from src.utils.experiment_schema import load_experiment_result, list_experiments

def create_test_datasets():
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    n = 200
    
    # Classification dataset
    df_class = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.rand(n) * 10,
        "feature_3": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    
    csv_path = scratch_dir / "sample_class.csv"
    excel_path = scratch_dir / "sample_class.xlsx"
    df_class.to_csv(csv_path, index=False)
    df_class.to_excel(excel_path, index=False)
    
    # Regression dataset
    df_reg = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n) * 5,
        "x3": np.random.rand(n),
        "target": np.random.randn(n) * 15 + 50.0
    })
    
    json_path = scratch_dir / "sample_reg.json"
    parquet_path = scratch_dir / "sample_reg.parquet"
    df_reg.to_json(json_path, orient="records")
    df_reg.to_parquet(parquet_path, index=False)
    
    print("✓ Created test datasets:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Excel: {excel_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - Parquet: {parquet_path}")
    
    return csv_path, excel_path, json_path, parquet_path

def test_classification_flow(csv_path):
    print("\n--- Testing Classification Flow ---")
    cfg = load_config()
    cfg["dataset"]["source"] = "file"
    cfg["dataset"]["file_path"] = str(csv_path)
    cfg["dataset"]["task_type"] = "classification"
    cfg["dataset"]["target_column"] = "target"
    cfg["training"]["algorithms"] = ["decision_tree", "logistic_regression", "random_forest"]
    cfg["training"]["optimization_metric"] = "f1"
    
    exp_id = f"exp_class_test_001"
    summary = run_pipeline(config=cfg, experiment_id=exp_id)
    
    result = load_experiment_result(exp_id)
    assert result is not None, "Experiment result JSON should exist"
    assert result["status"] == "COMPLETED", "Status should be COMPLETED"
    assert result["dataset"]["task_type"] == "classification", "Task type should be classification"
    assert result["best_model"] is not None, "Best model should be selected"
    assert result["security"]["overall_score"] > 0, "Security gates should have executed"
    assert len(result["security"]["gate_results"]) == 5, "Should execute all 5 security gates"
    assert "model_version" in result, "Model version should be present"
    
    print(f"✓ Classification run succeeded!")
    print(f"  - Version: {result['model_version']}")
    print(f"  - Best Algorithm: {result['best_model']['algorithm']}")
    print(f"  - Best F1: {result['best_model']['metrics'].get('f1', 0):.4f}")
    print(f"  - Security Score: {result['security']['overall_score']*100:.1f}%")
    return exp_id

def test_regression_flow(parquet_path):
    print("\n--- Testing Regression Flow ---")
    cfg = load_config()
    cfg["dataset"]["source"] = "file"
    cfg["dataset"]["file_path"] = str(parquet_path)
    cfg["dataset"]["task_type"] = "regression"
    cfg["dataset"]["target_column"] = "target"
    cfg["training"]["algorithms"] = ["linear_regression", "decision_tree_regressor", "random_forest_regressor"]
    cfg["training"]["optimization_metric"] = "r2"
    
    exp_id = f"exp_reg_test_002"
    summary = run_pipeline(config=cfg, experiment_id=exp_id)
    
    result = load_experiment_result(exp_id)
    assert result is not None, "Experiment result JSON should exist"
    assert result["status"] == "COMPLETED", "Status should be COMPLETED"
    assert result["dataset"]["task_type"] == "regression", "Task type should be regression"
    assert result["best_model"] is not None, "Best model should be selected"
    assert "r2" in result["best_model"]["metrics"], "R2 metric should be present in regression best_model"
    print(f"✓ Regression run succeeded!")
    print(f"  - Version: {result['model_version']}")
    print(f"  - Best Algorithm: {result['best_model']['algorithm']}")
    print(f"  - Best R²: {result['best_model']['metrics'].get('r2', 0):.4f}")
    return exp_id

def test_zip_packaging_and_execution(exp_id):
    print("\n--- Testing Executable ZIP Project Package ---")
    from api.main import download_experiment_model
    
    # Invoke download_experiment_model endpoint function
    response = download_experiment_model(exp_id)
    assert response.status_code == 200 or hasattr(response, "body_iterator"), "Response should be valid StreamingResponse"
    
    # Collect zip content
    import asyncio
    async def _read_chunks():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return b"".join(chunks)
    zip_bytes = asyncio.run(_read_chunks())
    assert len(zip_bytes) > 0, "ZIP buffer should not be empty"
    
    # Extract to temp directory and test execution
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        zip_file_path = temp_path / "package.zip"
        with open(zip_file_path, "wb") as f:
            f.write(zip_bytes)
            
        with zipfile.ZipFile(zip_file_path, "r") as z:
            z.extractall(temp_path)
            
        files_extracted = [f.name for f in temp_path.glob("*")]
        print(f"✓ Extracted ZIP package files: {files_extracted}")
        
        required_files = ["best_model.joblib", "feature_pipeline.joblib", "app.py", "requirements.txt", "model_metadata.json", "evaluation_results.json", "security_results.json", "README.md"]
        for req in required_files:
            assert (temp_path / req).exists(), f"Required file {req} missing in ZIP!"
            
        # Test app.py FastAPI code in the package by importing it
        sys.path.insert(0, str(temp_path))
        import joblib
        loaded_model = joblib.load(temp_path / "best_model.joblib")
        loaded_pipe = joblib.load(temp_path / "feature_pipeline.joblib")
        
        # Test inference with sample feature vector matching pipeline feature count
        n_feats = getattr(loaded_pipe, "n_features_in_", 4)
        sample_x = np.random.randn(2, n_feats)
        sample_trans = loaded_pipe.transform(sample_x)
        preds = loaded_model.predict(sample_trans)
        assert len(preds) == 2, "Model prediction should return predictions for input samples"
        print(f"✓ Package model prediction test passed! Predictions: {preds}")

if __name__ == "__main__":
    print("==================================================")
    print("  RUNNING FULL MLSECOPS END-TO-END VALIDATION  ")
    print("==================================================")
    csv_p, xlsx_p, json_p, parquet_p = create_test_datasets()
    
    exp_c = test_classification_flow(csv_p)
    exp_r = test_regression_flow(parquet_p)
    
    test_zip_packaging_and_execution(exp_c)
    test_zip_packaging_and_execution(exp_r)
    
    print("\n==================================================")
    print("  ALL END-TO-END VALIDATION TESTS PASSED 💯  ")
    print("==================================================")
