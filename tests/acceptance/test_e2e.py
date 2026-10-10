# tests/acceptance/test_e2e.py
# End-to-end acceptance tests for the automated-mlsecops API.
# Requires API server at localhost:8000.
# Start: uvicorn api.main:app --host 0.0.0.0 --port 8000
# Run  : py -m pytest tests/acceptance/ -v -o "addopts=" -s
from __future__ import annotations
import io, json, time, zipfile
import numpy as np
import pandas as pd
import pytest, requests

BASE = "http://localhost:8000"
TIMEOUT = 10

@pytest.fixture(scope="module")
def api():
    for _ in range(12):
        try:
            r = requests.get(f"{BASE}/health", timeout=3)
            if r.status_code == 200:
                return BASE
        except requests.ConnectionError:
            pass
        time.sleep(1)
    pytest.skip("API not reachable at localhost:8000")

def _clf_csv(n=120):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["label"] = y
    buf = io.BytesIO(); df.to_csv(buf, index=False); return buf.getvalue()

def _reg_csv(n=120):
    rng = np.random.default_rng(99)
    X = rng.standard_normal((n, 4))
    y = 3.0*X[:,0] - 2.0*X[:,1] + rng.normal(0, 0.1, n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
    df["price"] = y
    buf = io.BytesIO(); df.to_csv(buf, index=False); return buf.getvalue()

def _poll(base, exp_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{base}/experiments/{exp_id}", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        if data.get("status") in ("COMPLETED", "FAILED"):
            return data
        time.sleep(5)
    pytest.fail(f"{exp_id} did not finish in {timeout}s")

class TestHealthAndRoot:
    def test_root(self, api):
        r = requests.get(f"{api}/", timeout=TIMEOUT)
        assert r.status_code == 200
        b = r.json()
        assert "service" in b and "status" in b

    def test_health_200(self, api):
        r = requests.get(f"{api}/health", timeout=TIMEOUT)
        assert r.status_code == 200
        assert r.json()["status"] in ("healthy", "degraded")

    def test_health_keys(self, api):
        body = requests.get(f"{api}/health", timeout=TIMEOUT).json()
        for k in ("model_loaded", "feature_pipeline_loaded", "model_version"):
            assert k in body

class TestMetrics:
    def test_200(self, api):
        assert requests.get(f"{api}/metrics", timeout=TIMEOUT).status_code == 200

    def test_has_content(self, api):
        assert len(requests.get(f"{api}/metrics", timeout=TIMEOUT).text) > 0

class TestSecurityAudit:
    def test_200(self, api):
        assert requests.get(f"{api}/security/audit", timeout=TIMEOUT).status_code == 200

    def test_fields(self, api):
        b = requests.get(f"{api}/security/audit", timeout=TIMEOUT).json()
        assert "overall_status" in b
        assert b["overall_status"] in ("PASS", "FAIL")
        assert "dependency_audit" in b and "secrets_scan" in b

class TestPredict:
    """Predict tests — self-contained: uploads data, trains a model, then predicts."""

    N_FEATURES = 5  # must match _clf_csv()

    @pytest.fixture(scope="class")
    def loaded_model(self, api):
        """Upload 5-feature clf data, run a quick experiment, load its model."""
        # Upload dataset
        up = requests.post(
            f"{api}/data/upload",
            params={"target_column": "label", "task_type": "classification"},
            files={"file": ("predict_test.csv", _clf_csv(120), "text/csv")},
            timeout=30,
        )
        assert up.status_code == 200, f"Upload failed: {up.text}"

        # Trigger pipeline
        run = requests.post(
            f"{api}/experiments/run",
            json={"target_column": "label", "task_type": "classification",
                  "models_to_train": ["logistic_regression"], "opt_metric": "f1"},
            timeout=TIMEOUT,
        )
        assert run.status_code == 200
        exp_id = run.json()["experiment_id"]

        # Poll until complete
        result = _poll(api, exp_id, timeout=300)
        assert result["status"] == "COMPLETED", f"Pipeline failed: {result}"

        # Load the trained model into server memory
        load_r = requests.post(f"{api}/experiments/{exp_id}/model/load", timeout=TIMEOUT)
        assert load_r.status_code == 200, f"Model load failed: {load_r.text}"
        return exp_id

    def test_valid(self, api, loaded_model):
        feats = np.random.default_rng(1).standard_normal((1, self.N_FEATURES)).tolist()
        r = requests.post(f"{api}/predict", json={"features": feats}, timeout=TIMEOUT)
        assert r.status_code == 200, f"Predict failed ({r.status_code}): {r.text}"
        b = r.json()
        for k in ("predictions", "anomaly_detected", "model_version", "latency_ms"):
            assert k in b

    def test_empty_400(self, api, loaded_model):
        assert requests.post(f"{api}/predict", json={"features": []},
                             timeout=TIMEOUT).status_code == 400

    def test_multirow(self, api, loaded_model):
        feats = np.random.default_rng(7).standard_normal((3, self.N_FEATURES)).tolist()
        r = requests.post(f"{api}/predict", json={"features": feats}, timeout=TIMEOUT)
        assert r.status_code == 200, f"Predict failed ({r.status_code}): {r.text}"
        assert len(r.json()["predictions"]) == 3

class TestMonitoring:
    def test_stats_200(self, api):
        assert requests.get(f"{api}/monitoring/stats", timeout=TIMEOUT).status_code == 200

    def test_stats_sections(self, api):
        b = requests.get(f"{api}/monitoring/stats", timeout=TIMEOUT).json()
        for s in ("health","traffic","system","drift","performance_drift"):
            assert s in b

    def test_system_keys(self, api):
        b = requests.get(f"{api}/monitoring/stats", timeout=TIMEOUT).json()
        assert "cpu_percent" in b["system"] and "ram_used_gb" in b["system"]

    def test_ground_truth(self, api):
        # Check how many predictions are already logged
        stats = requests.get(f"{api}/monitoring/stats", timeout=TIMEOUT).json()
        total = stats.get("traffic", {}).get("total_requests", 0)
        # We can always safely submit ground truth for index 0 if any predictions exist
        # (TestPredict fixture runs before this and makes predictions)
        if total == 0:
            pytest.skip("No prediction logs yet — run TestPredict first or start server fresh")
        r = requests.post(f"{api}/monitoring/ground_truth",
            json={"prediction_indices": [0], "ground_truth": [1]}, timeout=TIMEOUT)
        assert r.status_code == 200
        assert r.json()["status"] == "success"
        assert r.json()["labels_received"] == 1

class TestExperimentsList:
    def test_200(self, api):
        assert requests.get(f"{api}/experiments", timeout=TIMEOUT).status_code == 200

    def test_key(self, api):
        b = requests.get(f"{api}/experiments", timeout=TIMEOUT).json()
        assert "experiments" in b and isinstance(b["experiments"], list)

class TestDataUpload:
    def test_clf(self, api):
        r = requests.post(f"{api}/data/upload",
            params={"target_column":"label","task_type":"classification"},
            files={"file":("clf.csv",_clf_csv(),"text/csv")}, timeout=30)
        assert r.status_code == 200
        b = r.json()
        assert b["status"]=="success" and b["file_size_bytes"]>0

    def test_reg(self, api):
        r = requests.post(f"{api}/data/upload",
            params={"target_column":"price","task_type":"regression"},
            files={"file":("reg.csv",_reg_csv(),"text/csv")}, timeout=30)
        assert r.status_code == 200
        assert r.json()["task_type"]=="regression"

    def test_bad_ext_400(self, api):
        r = requests.post(f"{api}/data/upload",
            params={"target_column":"label","task_type":"classification"},
            files={"file":("data.txt",b"a,b\n1,2","text/plain")}, timeout=10)
        assert r.status_code == 400

    def test_empty_400(self, api):
        r = requests.post(f"{api}/data/upload",
            params={"target_column":"label","task_type":"classification"},
            files={"file":("empty.csv",b"","text/csv")}, timeout=10)
        assert r.status_code == 400

class TestPipelineClassification:
    @pytest.fixture(scope="class")
    def clf_exp(self, api):
        requests.post(f"{api}/data/upload",
            params={"target_column":"label","task_type":"classification"},
            files={"file":("clf_e2e.csv",_clf_csv(150),"text/csv")}, timeout=30)
        r = requests.post(f"{api}/experiments/run",
            json={"target_column":"label","task_type":"classification",
                  "models_to_train":["logistic_regression"],"opt_metric":"f1"},
            timeout=TIMEOUT)
        assert r.status_code == 200
        return r.json()["experiment_id"]

    def test_queued(self, api, clf_exp):
        r = requests.get(f"{api}/experiments/{clf_exp}", timeout=TIMEOUT)
        assert r.status_code == 200
        assert r.json().get("status") in ("QUEUED","RUNNING","COMPLETED")

    def test_completes(self, api, clf_exp):
        res = _poll(api, clf_exp, 300)
        assert res["status"]=="COMPLETED", f"Pipeline failed: {res}"

    def test_has_best_model(self, api, clf_exp):
        res = _poll(api, clf_exp, 300)
        assert "best_model" in res and "algorithm" in res["best_model"]

    def test_has_security(self, api, clf_exp):
        res = _poll(api, clf_exp, 300)
        assert "security" in res

    def test_unknown_404(self, api):
        r = requests.get(f"{api}/experiments/nonexistent_xyz_999", timeout=TIMEOUT)
        assert r.status_code == 404

class TestPipelineRegression:
    @pytest.fixture(scope="class")
    def reg_exp(self, api):
        requests.post(f"{api}/data/upload",
            params={"target_column":"price","task_type":"regression"},
            files={"file":("reg_e2e.csv",_reg_csv(150),"text/csv")}, timeout=30)
        r = requests.post(f"{api}/experiments/run",
            json={"target_column":"price","task_type":"regression",
                  "models_to_train":["logistic_regression"],"opt_metric":"r2"},
            timeout=TIMEOUT)
        assert r.status_code == 200
        return r.json()["experiment_id"]

    def test_terminal_state(self, api, reg_exp):
        res = _poll(api, reg_exp, 300)
        assert res["status"] in ("COMPLETED","FAILED")

class TestModelDownload:
    @pytest.fixture(scope="class")
    def done_exp(self, api):
        requests.post(f"{api}/data/upload",
            params={"target_column":"label","task_type":"classification"},
            files={"file":("clf_dl.csv",_clf_csv(120),"text/csv")}, timeout=30)
        r = requests.post(f"{api}/experiments/run",
            json={"target_column":"label","task_type":"classification",
                  "models_to_train":["logistic_regression"],"opt_metric":"f1"},
            timeout=TIMEOUT)
        exp_id = r.json()["experiment_id"]
        res = _poll(api, exp_id, 300)
        if res["status"] != "COMPLETED":
            pytest.skip("Pipeline incomplete - skipping download tests")
        return exp_id

    def test_zip_magic(self, api, done_exp):
        r = requests.get(f"{api}/experiments/{done_exp}/model/download", timeout=30)
        assert r.status_code == 200
        assert r.content[:2] == b"PK"

    def test_zip_files(self, api, done_exp):
        r = requests.get(f"{api}/experiments/{done_exp}/model/download", timeout=30)
        names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
        for f in ("best_model.joblib","app.py","requirements.txt","README.md","model_metadata.json"):
            assert f in names

    def test_metadata_json(self, api, done_exp):
        r = requests.get(f"{api}/experiments/{done_exp}/model/download", timeout=30)
        meta = json.loads(zipfile.ZipFile(io.BytesIO(r.content)).read("model_metadata.json"))
        assert meta["experiment_id"] == done_exp

    def test_404(self, api):
        r = requests.get(f"{api}/experiments/bad_exp_xyz_000/model/download", timeout=TIMEOUT)
        assert r.status_code == 404
