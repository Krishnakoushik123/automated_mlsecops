import streamlit as st
import pandas as pd
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from api_client import APIClient
from io import BytesIO

st.set_page_config(page_title="Automated Secure MLOps", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- Modern Custom CSS ---
st.markdown("""
<style>
    :root {
        --primary-color: #4F46E5;
        --secondary-color: #10B981;
        --bg-color: #0F172A;
        --card-bg: #1E293B;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --border-color: #334155;
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
    }
    
    .metric-card {
        background-color: var(--card-bg);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-title {
        color: var(--text-secondary);
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: var(--text-primary);
        font-size: 32px;
        font-weight: 700;
        line-height: 1.2;
    }
    
    .status-pass { color: #10B981; }
    .status-fail { color: #EF4444; }
    .status-warn { color: #F59E0B; }
    .status-running { color: #3B82F6; }
    
    /* Progress Tracker Styles */
    .pipeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 40px 0;
        padding: 20px;
        background: var(--card-bg);
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
    
    .pipeline-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        position: relative;
    }
    
    .pipeline-step:not(:last-child)::after {
        content: '';
        position: absolute;
        top: 20px;
        right: -50%;
        width: 100%;
        height: 3px;
        background: var(--border-color);
        z-index: 1;
    }
    
    .pipeline-step.completed:not(:last-child)::after {
        background: var(--secondary-color);
    }
    
    .step-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--bg-color);
        border: 2px solid var(--border-color);
        color: var(--text-secondary);
        font-size: 18px;
        z-index: 2;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    
    .pipeline-step.completed .step-icon {
        background: var(--secondary-color);
        border-color: var(--secondary-color);
        color: white;
    }
    
    .pipeline-step.running .step-icon {
        background: var(--primary-color);
        border-color: var(--primary-color);
        color: white;
        animation: pulse 2s infinite;
    }
    
    .pipeline-step.failed .step-icon {
        background: #EF4444;
        border-color: #EF4444;
        color: white;
    }
    
    .step-label {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-secondary);
        text-align: center;
    }
    
    .pipeline-step.running .step-label { color: var(--primary-color); }
    .pipeline-step.completed .step-label { color: var(--secondary-color); }
    .pipeline-step.failed .step-label { color: #EF4444; }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(79, 70, 229, 0); }
        100% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0); }
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: opacity 0.2s;
        text-decoration: none;
        display: inline-block;
        text-align: center;
    }
    .btn-primary:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

api = APIClient()

def render_metric(title, value, color_class=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def render_pipeline_tracker(status):
    stages = [
        ("Data Validation", "📊"),
        ("Preprocessing", "⚙️"),
        ("Training", "🧠"),
        ("Evaluation", "📈"),
        ("Security", "🛡️"),
        ("Deployment", "🚀")
    ]
    
    # Map backend status to UI stages
    if status == "QUEUED":
        current_stage = 0
    elif status == "RUNNING":
        current_stage = 2 # Simplify for now, animation shows it's working
    elif status == "COMPLETED":
        current_stage = len(stages)
    elif status == "FAILED":
        current_stage = -1
    else:
        current_stage = 0

    html = '<div class="pipeline-container">'
    for i, (label, icon) in enumerate(stages):
        state = ""
        if current_stage == -1:
            state = "failed" if i == 2 else "completed" if i < 2 else ""
        elif i < current_stage:
            state = "completed"
        elif i == current_stage:
            state = "running"
            
        html += f"""
        <div class="pipeline-step {state}">
            <div class="step-icon">{icon if state != 'completed' else '✓'}</div>
            <div class="step-label">{label}</div>
        </div>
        """
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🛡️ Secure MLOps")
    st.markdown("Automated pipeline for training, securing, and deploying ML models.")
    st.markdown("---")
    page = st.radio("Navigation", ["Dashboard", "New Experiment", "Results", "Monitoring"], label_visibility="collapsed")
    st.markdown("---")
    
    if st.button("🔄 Check API Status", use_container_width=True):
        health = api.get_health()
        if health:
            st.success("API Online")
        else:
            st.error("API Offline")

if "selected_experiment_id" not in st.session_state:
    st.session_state["selected_experiment_id"] = "latest"

# --- Dashboard ---
if page == "Dashboard":
    st.title("System Overview")
    
    exps = api.get_experiments()
    if not exps:
        st.info("No experiments found. Start a new experiment to begin.")
    else:
        total = len(exps)
        success = sum(1 for e in exps if e.get("status") == "COMPLETED")
        failed = sum(1 for e in exps if e.get("status") == "FAILED")
        
        c1, c2, c3 = st.columns(3)
        with c1: render_metric("Total Experiments", total)
        with c2: render_metric("Successful Deployments", success, "status-pass")
        with c3: render_metric("Failed Runs", failed, "status-fail")
        
        st.markdown("### Recent Activity")
        df_exps = pd.DataFrame(exps)
        if not df_exps.empty:
            cols = ["experiment_id", "status"]
            if "dataset" in df_exps.columns:
                df_exps["dataset_name"] = df_exps["dataset"].apply(lambda x: x.get("name") if isinstance(x, dict) else "Unknown")
                cols.append("dataset_name")
            st.dataframe(df_exps[cols].head(10), use_container_width=True)

# --- New Experiment ---
elif page == "New Experiment":
    st.title("New Automation Run")
    st.markdown("Upload your dataset and configure the ML pipeline.")
    
    with st.container():
        st.markdown("### 1. Upload Dataset")
        uploaded_file = st.file_uploader("Select dataset", type=["csv", "xlsx", "xls", "json", "parquet"], help="Supported formats: CSV, Excel, JSON, Parquet")
        
        if uploaded_file:
            try:
                ext = uploaded_file.name.split(".")[-1].lower()
                if ext == "csv": df = pd.read_csv(uploaded_file)
                elif ext in ["xlsx", "xls"]: df = pd.read_excel(uploaded_file)
                elif ext == "json": df = pd.read_json(uploaded_file)
                elif ext == "parquet": df = pd.read_parquet(uploaded_file)
                
                st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
                st.dataframe(df.head(), use_container_width=True)
                
                st.markdown("### 2. Configuration")
                c1, c2 = st.columns(2)
                
                with c1:
                    target_col = st.selectbox("Target Column", df.columns.tolist(), index=len(df.columns)-1)
                    
                    # Auto-detect task type
                    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
                    n_unique = df[target_col].nunique()
                    
                    if is_numeric and n_unique > 15:
                        task_type = "regression"
                    else:
                        task_type = "classification"
                        
                    task_type = st.selectbox("Task Type", ["classification", "regression"], index=0 if task_type=="classification" else 1)
                
                with c2:
                    if task_type == "classification":
                        metrics = ["f1", "roc_auc", "accuracy", "precision", "recall"]
                        algos = {
                            "Logistic Regression": "logistic_regression",
                            "Random Forest": "random_forest",
                            "Gradient Boosting": "gradient_boosting",
                            "Decision Tree": "decision_tree",
                            "SVM": "svm",
                            "KNN": "knn",
                            "XGBoost": "xgboost"
                        }
                    else:
                        metrics = ["rmse", "mae", "r2"]
                        algos = {
                            "Linear Regression": "linear_regression",
                            "Random Forest Regressor": "random_forest_regressor",
                            "Gradient Boosting Regressor": "gradient_boosting_regressor",
                            "Decision Tree Regressor": "decision_tree_regressor",
                            "SVM Regressor": "svm_regressor",
                            "KNN Regressor": "knn_regressor",
                            "XGBoost Regressor": "xgboost_regressor"
                        }
                    
                    opt_metric = st.selectbox("Optimization Metric", metrics)
                    selected_algos_names = st.multiselect("Algorithms to Train", list(algos.keys()), default=list(algos.keys())[:3])
                    selected_algos = [algos[name] for name in selected_algos_names]
                
                if st.button("🚀 Start Automation Pipeline", type="primary", use_container_width=True):
                    if not selected_algos:
                        st.error("Please select at least one algorithm.")
                    else:
                        with st.spinner("Initializing MLSecOps Pipeline..."):
                            uploaded_file.seek(0)
                            res = api.upload_dataset(uploaded_file.name, uploaded_file.getvalue())
                            
                            if res and res.get("status") == "success":
                                run_res = api.run_pipeline(
                                    target_column=target_col,
                                    opt_metric=opt_metric,
                                    models_to_train=selected_algos,
                                    task_type=task_type
                                )
                                
                                if run_res and "experiment_id" in run_res:
                                    st.session_state["polling_exp_id"] = run_res["experiment_id"]
                                    st.rerun()
                                else:
                                    st.error("Failed to start pipeline backend.")
                            else:
                                st.error("Failed to upload dataset to backend.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # Polling logic
    if "polling_exp_id" in st.session_state:
        exp_id = st.session_state["polling_exp_id"]
        st.markdown("---")
        st.markdown(f"### Pipeline Status: `{exp_id}`")
        
        status_placeholder = st.empty()
        tracker_placeholder = st.empty()
        
        while True:
            exp_data = api.get_experiment(exp_id)
            if not exp_data:
                status_placeholder.warning("Waiting for pipeline to initialize...")
                time.sleep(2)
                continue
                
            status = exp_data.get("status", "UNKNOWN")
            
            with tracker_placeholder:
                render_pipeline_tracker(status)
                
            if status in ["COMPLETED", "FAILED"]:
                st.session_state["selected_experiment_id"] = exp_id
                if status == "COMPLETED":
                    status_placeholder.success("✨ Pipeline completed successfully! View full details in the Results tab.")
                else:
                    status_placeholder.error(f"❌ Pipeline failed: {exp_data.get('error', 'Unknown error')}")
                del st.session_state["polling_exp_id"]
                break
                
            status_placeholder.info("⏳ Processing... please wait.")
            time.sleep(2.5)

# --- Results ---
elif page == "Results":
    st.title("Experiment Results")
    
    exps = api.get_experiments()
    if not exps:
        st.info("No completed experiments available.")
    else:
        exp_options = [e["experiment_id"] for e in exps if "experiment_id" in e]
        if not exp_options:
            st.warning("No valid experiments found.")
        else:
            default_idx = 0
            if st.session_state["selected_experiment_id"] in exp_options:
                default_idx = exp_options.index(st.session_state["selected_experiment_id"])
                
            selected = st.selectbox("Select Run", exp_options, index=default_idx)
            st.session_state["selected_experiment_id"] = selected
            
            exp = api.get_experiment(selected)
            
            if not exp:
                st.error("Failed to load experiment data.")
            elif exp.get("status") == "FAILED":
                st.error("This experiment failed.")
                st.code(exp.get("error_message", "Unknown error"))
            elif exp.get("status") in ["QUEUED", "RUNNING"]:
                st.info("Experiment is currently running. Please wait.")
            else:
                # Main Results View
                st.markdown("---")
                c1, c2 = st.columns([2, 1])
                
                ds = exp.get("dataset", {})
                bm = exp.get("best_model", {})
                sec = exp.get("security", {})
                res = exp.get("resources", {})
                task_type = ds.get("task_type", "classification")
                
                with c1:
                    st.markdown("### 🏆 Best Model")
                    st.markdown(f"**Algorithm:** `{bm.get('algorithm', 'N/A')}`")
                    
                    st.markdown("### 📊 Performance")
                    rm = exp.get("research_metrics", {})
                    metrics_cols = st.columns(4)
                    
                    if task_type == "classification":
                        metrics_cols[0].metric("F1 Score", f"{rm.get('f1_score', 0):.4f}")
                        metrics_cols[1].metric("Accuracy", f"{rm.get('accuracy', 0):.4f}")
                        metrics_cols[2].metric("ROC AUC", f"{rm.get('roc_auc', 0):.4f}")
                    else:
                        metrics_cols[0].metric("RMSE", f"{rm.get('rmse', 0):.4f}")
                        metrics_cols[1].metric("MAE", f"{rm.get('mae', 0):.4f}")
                        metrics_cols[2].metric("R² Score", f"{rm.get('r2_score', 0):.4f}")
                        
                    metrics_cols[3].metric("Inference Latency", f"{rm.get('inference_latency_p95_ms', 0):.2f} ms")
                    
                    # Model Comparison Chart
                    models = exp.get("models", {})
                    if models:
                        st.markdown("### 📈 Model Comparison")
                        comp_data = []
                        for algo, data in models.items():
                            m = data.get("metrics", {})
                            if task_type == "classification":
                                comp_data.append({
                                    "Algorithm": algo,
                                    "Score": m.get("test_f1", 0),
                                    "Metric": "F1 Score"
                                })
                            else:
                                comp_data.append({
                                    "Algorithm": algo,
                                    "Score": m.get("test_r2", 0),
                                    "Metric": "R² Score"
                                })
                        df_comp = pd.DataFrame(comp_data)
                        fig = px.bar(df_comp, x="Algorithm", y="Score", color="Score", color_continuous_scale="Viridis")
                        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                        st.plotly_chart(fig, use_container_width=True)

                with c2:
                    st.markdown("### 📦 Deployment")
                    download_url = api.get_model_download_url(selected)
                    st.markdown(f"""
                    <div style="background: var(--card-bg); padding: 20px; border-radius: 12px; border: 1px solid var(--border-color); text-align: center;">
                        <h4 style="margin-top: 0;">Deployable Project</h4>
                        <p style="color: var(--text-secondary); font-size: 14px;">Includes trained model, preprocessing pipeline, requirements, and FastAPI server.</p>
                        <a href="{download_url}" class="btn-primary" style="width: 100%;" download>📥 Download ZIP Package</a>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("### 🛡️ Security Check")
                    sec_score = sec.get("overall_score", 0) * 100
                    passed = sec.get("overall_passed", False)
                    color = "status-pass" if passed else "status-fail"
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom: 0;">
                        <div class="metric-title">Security Score</div>
                        <div class="metric-value {color}">{sec_score:.1f}%</div>
                        <div style="margin-top: 10px; font-weight: 600; color: {'#10B981' if passed else '#EF4444'}">
                            {'✅ Passed all gates' if passed else '❌ Failed security gates'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("### ⚙️ Resources")
                    st.markdown(f"- **Training Time:** {res.get('pipeline_duration_sec', 0):.1f}s")
                    st.markdown(f"- **CPU Count:** {res.get('cpu_count', 'N/A')}")
                    st.markdown(f"- **RAM:** {res.get('ram_available_gb', 0):.1f} GB")

# --- Monitoring ---
elif page == "Monitoring":
    st.title("API Monitoring")
    st.markdown("Live metrics from the backend inference service.")
    
    stats = api.get_monitoring_stats()
    if stats:
        c1, c2, c3 = st.columns(3)
        with c1: render_metric("Total Requests", stats.get("inference_requests", 0))
        with c2: render_metric("Anomalies Blocked", stats.get("anomalies_detected", 0), "status-warn")
        with c3: render_metric("Avg Latency", f"{stats.get('avg_latency_ms', 0):.2f} ms")
        
        st.markdown("### Prometheus Metrics Dump")
        metrics = api.get_metrics()
        if metrics:
            with st.expander("View Raw Data"):
                st.code(metrics, language="text")
    else:
        st.warning("Monitoring service unavailable. Ensure backend is running.")
