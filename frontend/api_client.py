import os
import requests
import time
from typing import Dict, Any, List, Optional

class APIClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("API_URL", "http://localhost:8000")
        
    def _get(self, endpoint: str) -> Optional[Any]:
        try:
            response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"APIClient GET {endpoint} Error: {e}")
            return None
            
    def _post(self, endpoint: str, data: Dict = None, files: Dict = None) -> Optional[Any]:
        try:
            if files:
                response = requests.post(f"{self.base_url}{endpoint}", files=files, timeout=30)
            else:
                response = requests.post(f"{self.base_url}{endpoint}", json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"APIClient POST {endpoint} Error: {e}")
            return None

    def get_health(self) -> Optional[Dict[str, Any]]:
        return self._get("/health")

    def get_experiments(self) -> List[Dict[str, Any]]:
        res = self._get("/experiments")
        if res and "experiments" in res:
            return res["experiments"]
        return []

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return self._get(f"/experiments/{experiment_id}")

    def run_pipeline(self, target_column: str = "target", opt_metric: str = "f1", models_to_train: List[str] = None, task_type: str = "classification") -> Optional[Dict[str, Any]]:
        if models_to_train is None:
            models_to_train = ["logistic_regression", "random_forest", "gradient_boosting"]
        data = {
            "target_column": target_column,
            "opt_metric": opt_metric,
            "models_to_train": models_to_train,
            "task_type": task_type
        }
        return self._post("/experiments/run", data=data)

    def upload_dataset(self, file_name: str, file_bytes: bytes) -> Optional[Dict[str, Any]]:
        files = {"file": (file_name, file_bytes, "application/octet-stream")}
        return self._post("/data/upload", files=files)

    def get_metrics(self) -> Optional[str]:
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"APIClient GET /metrics Error: {e}")
            return None

    def get_monitoring_stats(self) -> Optional[Dict[str, Any]]:
        return self._get("/monitoring/stats")

    def load_model(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return self._post(f"/experiments/{experiment_id}/model/load")
        
    def get_model_download_url(self, experiment_id: str) -> str:
        return f"{self.base_url}/experiments/{experiment_id}/model/download"
