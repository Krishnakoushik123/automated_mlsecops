import streamlit as st
import pandas as pd
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from api_client import APIClient
from io import BytesIO

st.set_page_config(page_title="Automated Secure MLOps", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- Modern Custom CSS with Project-Themed Background Animation ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary-color: #6366F1;
        --primary-hover: #4F46E5;
        --secondary-color: #10B981;
        --bg-color: #0F172A;
        --card-bg: rgba(30, 41, 59, 0.75);
        --card-border: #334155;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #0B0F19;
        background-image: 
            radial-gradient(at 10% 10%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
            radial-gradient(at 90% 90%, rgba(16, 185, 129, 0.12) 0px, transparent 50%),
            radial-gradient(at 50% 50%, rgba(15, 23, 42, 0.8) 0px, transparent 100%);
        background-attachment: fixed;
        color: var(--text-primary);
    }

    /* Subtle Flowing Network Nodes / Particle Background Animation */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none;
        background: 
            radial-gradient(2px 2px at 20px 30px, rgba(255, 255, 255, 0.15), rgba(0,0,0,0)),
            radial-gradient(2px 2px at 40px 70px, rgba(99, 102, 241, 0.25), rgba(0,0,0,0)),
            radial-gradient(3px 3px at 80px 120px, rgba(16, 185, 129, 0.2), rgba(0,0,0,0)),
            radial-gradient(2px 2px at 150px 180px, rgba(255, 255, 255, 0.15), rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 200px 200px;
        animation: floatParticles 40s linear infinite;
        z-index: 0;
        opacity: 0.6;
    }

    @keyframes floatParticles {
        0% { background-position: 0 0; }
        100% { background-position: 200px 400px; }
    }
    
    .metric-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        padding: 20px 24px;
        border-radius: 12px;
        border: 1px solid var(--card-border);
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .metric-title {
        color: var(--text-secondary);
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: var(--text-primary);
        font-size: 28px;
        font-weight: 700;
        line-height: 1.2;
    }
    
    .status-pass { color: #10B981; }
    .status-fail { color: #EF4444; }
    .status-warn { color: #F59E0B; }
    .status-running { color: #3B82F6; }
    
    /* Security Gate Badge */
    .gate-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .gate-pass { background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .gate-fail { background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* Progress Tracker Styles */
    .pipeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 30px 0;
        padding: 24px;
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        border: 1px solid var(--card-border);
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
        background: var(--card-border);
        z-index: 1;
    }
    
    .pipeline-step.completed:not(:last-child)::after {
        background: var(--secondary-color);
    }
    
    .step-icon {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #0F172A;
        border: 2px solid var(--card-border);
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
        0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.5); }
        70% { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
        100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #6366F1 0%, #3B82F6 100%);
        color: white !important;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.15s, opacity 0.2s;
        text-decoration: none !important;
        display: block;
        text-align: center;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
    .btn-primary:hover {
        opacity: 0.95;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

api = APIClient()

def render_metric(title, value, color_class=""):
    card_html = f'<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value {color_class}">{value}</div></div>'
    st.markdown(card_html, unsafe_allow_html=True)

def render_pipeline_tracker(current_stage_name="Upload", status="RUNNING"):
    """Renders the authentic 7-stage MLSecOps pipeline progress tracker.
    Stages: Upload → Quality Check → Cleaning → Training → Evaluation → Security → Packaging
    """
    stages = [
        ("Upload", "📤"),
        ("Quality Check", "🔍"),
        ("Cleaning", "🧹"),
        ("Training", "🧠"),
        ("Evaluation", "📈"),
        ("Security", "🛡️"),
        ("Packaging", "📦"),
    ]
    
    cur_lower = (str(current_stage_name) or "").lower().replace("_", " ")
    match_idx = 0
    for idx, (s_name, _) in enumerate(stages):
        if s_name.lower() in cur_lower or cur_lower in s_name.lower():
            match_idx = idx
            break

    if status == "COMPLETED":
        current_idx = len(stages)
    elif status == "FAILED":
        current_idx = match_idx
    elif status == "QUEUED":
        current_idx = 0
    else:
        current_idx = match_idx

    html_parts = ['<div class="pipeline-container">']
    for i, (label, icon) in enumerate(stages):
        if status == "FAILED" and i == current_idx:
            state = "failed"
            step_icon = "✗"
        elif i < current_idx:
            state = "completed"
            step_icon = "✓"
        elif i == current_idx and status == "RUNNING":
            state = "running"
            step_icon = icon
        else:
            state = ""
            step_icon = icon

        html_parts.append(
            f'<div class="pipeline-step {state}">'
            f'<div class="step-icon">{step_icon}</div>'
            f'<div class="step-label">{label}</div>'
            f'</div>'
        )
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🛡️ Secure MLOps")
    st.markdown("Automated Pipeline for Training, Security Verification & Executable Deployment.")
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
    st.markdown("Real-time pipeline monitoring and historical runs dashboard.")
    
    exps = api.get_experiments()
    if not exps:
        st.info("No experiments found. Start a new experiment to begin.")
    else:
        total = len(exps)
        success = sum(1 for e in exps if e.get("status") == "COMPLETED")
        failed = sum(1 for e in exps if e.get("status") == "FAILED")
        sec_avg = sum(e.get("security_score", 0) for e in exps) / max(1, total) * 100
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: render_metric("Total Experiments", total)
        with c2: render_metric("Successful Runs", success, "status-pass")
        with c3: render_metric("Failed Runs", failed, "status-fail" if failed > 0 else "")
        with c4: render_metric("Avg Security Compliance", f"{sec_avg:.1f}%", "status-pass")
        
        st.markdown("### 📋 Recent Automation Runs")
        df_exps = pd.DataFrame(exps)
        if not df_exps.empty:
            df_display = pd.DataFrame()
            df_display["Experiment ID"] = df_exps["experiment_id"]
            # Replace Version column with Dataset ID; retain model versioning separately
            df_display["Dataset ID"] = df_exps.get("dataset_id", "ds_default")
            df_display["Dataset File"] = df_exps.get("dataset_name", "dataset.csv")
            df_display["Task Type"] = df_exps.get("task_type", "classification")
            df_display["Best Algorithm"] = df_exps.get("best_model", "N/A")
            df_display["Status"] = df_exps["status"]
            df_display["Created At"] = df_exps.get("created_at", "N/A")
            
            st.dataframe(df_display.head(10), use_container_width=True)

# --- New Experiment ---
elif page == "New Experiment":
    st.title("New Automation Run")
    st.markdown("Upload your dataset and configure the automated MLSecOps pipeline.")
    
    with st.container():
        st.markdown("### 1. Upload Dataset")
        uploaded_file = st.file_uploader("Select dataset file", type=["csv", "xlsx", "xls", "json", "parquet"], help="Supported formats: CSV, Excel (.xlsx, .xls), JSON, Parquet")
        
        if uploaded_file:
            try:
                ext = uploaded_file.name.split(".")[-1].lower()
                if ext == "csv": df = pd.read_csv(uploaded_file)
                elif ext in ["xlsx", "xls"]: df = pd.read_excel(uploaded_file)
                elif ext == "json": df = pd.read_json(uploaded_file)
                elif ext == "parquet": df = pd.read_parquet(uploaded_file)
                
                st.success(f"✓ Dataset '{uploaded_file.name}' loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
                st.dataframe(df.head(), use_container_width=True)
                
                st.markdown("### 2. Configure Automation Target & Algorithms")
                c1, c2 = st.columns(2)
                
                with c1:
                    target_col = st.selectbox("Target Column", df.columns.tolist(), index=len(df.columns)-1)
                    
                    # Auto-detect task type
                    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
                    n_unique = df[target_col].nunique()
                    
                    if is_numeric and n_unique > 15:
                        auto_task = "regression"
                    else:
                        auto_task = "classification"
                        
                    task_type = st.selectbox("Task Type", ["classification", "regression"], index=0 if auto_task=="classification" else 1)
                
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
                    selected_algos_names = st.multiselect("Algorithms to Train & Evaluate", list(algos.keys()), default=list(algos.keys())[:3])
                    selected_algos = [algos[name] for name in selected_algos_names]
                
                is_active = "polling_exp_id" in st.session_state
                if st.button("🚀 Start End-to-End Automation Pipeline", type="primary", use_container_width=True, disabled=is_active):
                    if not selected_algos:
                        st.error("Please select at least one algorithm.")
                    else:
                        with st.spinner("Uploading dataset and triggering MLSecOps Pipeline..."):
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
                                    st.error("Failed to start pipeline on backend.")
                            else:
                                st.error("Failed to upload dataset file to backend.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # Polling logic for ongoing run with genuine backend progress
    if "polling_exp_id" in st.session_state:
        exp_id = st.session_state["polling_exp_id"]
        st.markdown("---")
        st.markdown(f"### Pipeline Execution Progress: `{exp_id}`")
        
        status_placeholder = st.empty()
        tracker_placeholder = st.empty()
        
        while True:
            exp_data = api.get_experiment(exp_id)
            if not exp_data:
                status_placeholder.warning("Waiting for pipeline orchestrator to initialize...")
                time.sleep(1.5)
                continue
                
            status = exp_data.get("status", "UNKNOWN")
            stage = exp_data.get("stage", "Upload")
            stage_details = exp_data.get("stage_details", "")
            
            with tracker_placeholder:
                render_pipeline_tracker(stage, status)
                
            if status in ["COMPLETED", "FAILED"]:
                st.session_state["selected_experiment_id"] = exp_id
                if status == "COMPLETED":
                    render_pipeline_tracker("Packaging", "COMPLETED")
                    status_placeholder.success("✨ Pipeline completed successfully! All 7 stages passed. View comprehensive results in the Results tab.")
                else:
                    render_pipeline_tracker(stage, "FAILED")
                    status_placeholder.error(f"❌ Pipeline failed at stage '{stage}': {exp_data.get('error', exp_data.get('error_message', 'Unknown error'))}")
                del st.session_state["polling_exp_id"]
                break
                
            status_placeholder.info(f"⏳ **Active Stage: {stage}** — {stage_details or 'Executing MLSecOps pipeline...'}")
            time.sleep(2.0)

# --- Results ---
elif page == "Results":
    st.title("Experiment Results & Governance")
    
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
                
            selected = st.selectbox("Select Experiment Run", exp_options, index=default_idx)
            st.session_state["selected_experiment_id"] = selected
            
            exp = api.get_experiment(selected)
            
            if not exp:
                st.error("Failed to load experiment data.")
            elif exp.get("status") == "FAILED":
                st.error("This experiment failed.")
                st.code(exp.get("error_message", "Unknown error"))
            elif exp.get("status") in ["QUEUED", "RUNNING"]:
                st.info("Experiment is currently executing. Please wait.")
            else:
                ds = exp.get("dataset", {})
                bm = exp.get("best_model", {})
                sec = exp.get("security", {})
                res = exp.get("resources", {})
                rm = exp.get("research_metrics", {})
                models = exp.get("models", {})
                task_type = ds.get("task_type", rm.get("task_type", "classification"))
                model_ver = exp.get("model_version") or exp.get("deployment", {}).get("model_version", "v1")
                
                # --- Top Performance Metrics Summary ---
                st.markdown("---")
                c1, c2, c3, c4, c5 = st.columns(5)
                
                with c1:
                    render_metric("Model Version", model_ver)
                with c2:
                    render_metric("Selected Algorithm", bm.get("algorithm", "N/A"))
                with c3:
                    if task_type == "classification":
                        render_metric("Primary F1 Score", f"{rm.get('f1_score', 0):.4f}", "status-pass")
                    else:
                        render_metric("Primary R² Score", f"{rm.get('r2_score', 0):.4f}", "status-pass")
                with c4:
                    sec_score = sec.get("overall_score", 0) * 100
                    sec_pass = sec.get("overall_passed", False)
                    render_metric("Security Score", f"{sec_score:.1f}%", "status-pass" if sec_pass else "status-fail")
                with c5:
                    render_metric("Inference Latency P95", f"{rm.get('latency_p95_ms', 0):.2f} ms")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                # --- Results Page Responsive Grid (Requirement 1) ---
                st.markdown("### 📊 Performance Analysis Dashboard")
                g1, g2 = st.columns(2)
                
                with g1:
                    # Chart 1: Model Performance Comparison
                    if models:
                        comp_data = []
                        for algo, data in models.items():
                            m = data.get("metrics", {})
                            if task_type == "classification":
                                comp_data.append({"Algorithm": algo, "F1 Score": m.get("f1", 0), "Accuracy": m.get("accuracy", 0)})
                            else:
                                comp_data.append({"Algorithm": algo, "R² Score": m.get("r2", 0), "MAE": m.get("mae", 0)})
                        df_comp = pd.DataFrame(comp_data)
                        main_metric = "F1 Score" if task_type == "classification" else "R² Score"
                        fig1 = px.bar(
                            df_comp, x="Algorithm", y=main_metric, color=main_metric,
                            title=f"Algorithm Comparison ({main_metric})",
                            color_continuous_scale="Tealgrn" if task_type == "classification" else "Viridis"
                        )
                        fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=320)
                        st.plotly_chart(fig1, use_container_width=True)
                        
                with g2:
                    # Chart 2: Inference Latency Comparison
                    if models:
                        lat_data = []
                        for algo, data in models.items():
                            m = data.get("metrics", {})
                            lat_data.append({"Algorithm": algo, "P50 (ms)": m.get("latency_p50_ms", 0), "P95 (ms)": m.get("latency_p95_ms", 0), "P99 (ms)": m.get("latency_p99_ms", 0)})
                        df_lat = pd.DataFrame(lat_data)
                        fig2 = px.bar(
                            df_lat, x="Algorithm", y=["P50 (ms)", "P95 (ms)", "P99 (ms)"],
                            title="Inference Latency Breakdown (ms)", barmode="group",
                            color_discrete_sequence=["#10B981", "#6366F1", "#F59E0B"]
                        )
                        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=320)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                g3, g4 = st.columns(2)
                
                with g3:
                    # Chart 3: Detailed Metrics Matrix
                    if models:
                        det_data = []
                        for algo, data in models.items():
                            m = data.get("metrics", {})
                            if task_type == "classification":
                                for met_name in ["accuracy", "precision", "recall", "f1"]:
                                    det_data.append({"Algorithm": algo, "Metric": met_name.upper(), "Value": m.get(met_name, 0)})
                            else:
                                for met_name in ["rmse", "mae", "r2"]:
                                    det_data.append({"Algorithm": algo, "Metric": met_name.upper(), "Value": m.get(met_name, 0)})
                        df_det = pd.DataFrame(det_data)
                        fig3 = px.bar(
                            df_det, x="Metric", y="Value", color="Algorithm",
                            title="Detailed Evaluation Metrics per Algorithm", barmode="group",
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=320)
                        st.plotly_chart(fig3, use_container_width=True)
                        
                with g4:
                    # Chart 4: Training Compute Duration
                    if models:
                        dur_data = []
                        for algo, data in models.items():
                            m = data.get("metrics", {})
                            dur_data.append({"Algorithm": algo, "Training Time (s)": m.get("training_time_sec", 0)})
                        df_dur = pd.DataFrame(dur_data)
                        fig4 = px.bar(
                            df_dur, x="Algorithm", y="Training Time (s)", color="Training Time (s)",
                            title="Training Compute Duration (Seconds)", color_continuous_scale="Magma"
                        )
                        fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=320)
                        st.plotly_chart(fig4, use_container_width=True)

                st.markdown("---")
                
                # --- Security Audit Results (Requirement 6) ---
                st.markdown("### 🛡️ Real-Time Security Gates Audit")
                st.markdown("Comprehensive security scans executed prior to model packaging:")
                
                gate_results = sec.get("gate_results", [])
                if not gate_results:
                    st.warning("No security gate details found for this experiment run.")
                else:
                    sec_cols = st.columns(len(gate_results))
                    for i, g in enumerate(gate_results):
                        gate_name = g.get("gate", f"Gate {i+1}").replace("_", " ").title()
                        passed = g.get("passed", False)
                        score = g.get("score", 0.0) * 100
                        dur = g.get("duration_seconds", 0.0)
                        
                        with sec_cols[i]:
                            status_html = f'<span class="gate-badge gate-pass">✓ PASS</span>' if passed else f'<span class="gate-badge gate-fail">✗ FAIL</span>'
                            st.markdown(f"""
                            <div class="metric-card" style="min-height: 150px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div class="metric-title">{gate_name}</div>
                                    {status_html}
                                </div>
                                <div class="metric-value" style="font-size: 24px; margin-top: 10px;">{score:.1f}%</div>
                                <div style="color: var(--text-secondary); font-size: 12px; margin-top: 6px;">Duration: {dur:.2f}s</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander(f"Details ({gate_name})"):
                                st.json(g.get("details", {}))
                                
                st.markdown("---")
                
                # --- Deployment & Executable Project Download (Requirement 4 & 5) ---
                d1, d2 = st.columns([1, 1])
                
                with d1:
                    st.markdown("### 📦 Executable Deployment Package")
                    download_url = api.get_model_download_url(selected)
                    st.markdown(f"""
                    <div style="background: var(--card-bg); padding: 24px; border-radius: 12px; border: 1px solid var(--card-border);">
                        <h4 style="margin-top: 0; color: var(--text-primary);">Deployable Production ZIP Package</h4>
                        <p style="color: var(--text-secondary); font-size: 14px;">Contains trained <code>best_model.joblib</code>, <code>feature_pipeline.joblib</code>, executable FastAPI <code>app.py</code>, requirements, evaluation results, security audit report, and model metadata.</p>
                        <a href="{download_url}" class="btn-primary" download>📥 Download Executable ZIP Package ({model_ver})</a>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with d2:
                    st.markdown("### ⚙️ Resource & Dataset Metrics")
                    dq = exp.get("data_quality", {})
                    cleaning = dq.get("cleaning", {})
                    actions = cleaning.get("actions", [])
                    st.markdown(f"""
                    <div style="background: var(--card-bg); padding: 24px; border-radius: 12px; border: 1px solid var(--card-border);">
                        <ul style="list-style-type: none; padding-left: 0; margin-bottom: 0;">
                            <li style="margin-bottom: 10px;">🏷️ <strong>Dataset ID:</strong> <code>{ds.get('dataset_id', 'ds_default')}</code></li>
                            <li style="margin-bottom: 10px;">📁 <strong>Dataset Filename:</strong> <code>{ds.get('file_name', ds.get('name', 'dataset.csv'))}</code></li>
                            <li style="margin-bottom: 10px;">🎯 <strong>Target Column:</strong> <code>{ds.get('target_column', ds.get('name', 'target'))}</code></li>
                            <li style="margin-bottom: 10px;">📊 <strong>Cleaned Shape:</strong> {ds.get('n_samples', 'N/A')} rows × {ds.get('n_features', 'N/A')} features</li>
                            <li style="margin-bottom: 10px;">⚡ <strong>Total Execution Time:</strong> {res.get('pipeline_execution_time_sec', 0):.2f} seconds</li>
                            <li style="margin-bottom: 0;">🧠 <strong>Memory Allocated:</strong> {res.get('ram_used_gb', 0)} GB / {res.get('ram_total_gb', 0)} GB ({res.get('ram_usage_percent', 0)}%)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    if actions:
                        with st.expander(f"🧹 Data Cleaning Report ({len(actions)} actions applied)"):
                            for act in actions:
                                st.markdown(f"- ✓ {act}")
                            if "before" in cleaning and "after" in cleaning:
                                b = cleaning["before"]
                                a = cleaning["after"]
                                st.table(pd.DataFrame({
                                    "Metric": ["Rows", "Features", "Duplicates", "Missing Cells", "Missing Ratio"],
                                    "Before Cleaning": [b.get("rows"), b.get("features"), b.get("duplicate_rows"), b.get("total_missing"), f"{b.get('missing_ratio', 0):.2%}"],
                                    "After Cleaning": [a.get("rows"), a.get("features"), a.get("duplicate_rows"), a.get("total_missing"), f"{a.get('missing_ratio', 0):.2%}"]
                                }))

# --- Monitoring ---
elif page == "Monitoring":
    st.title("🔍 API & System Monitoring")
    st.markdown("Live operational metrics collected from **actual prediction requests** and system measurements.")

    # Auto-refresh control
    col_refresh, col_interval = st.columns([2, 1])
    with col_refresh:
        auto_refresh = st.checkbox("⚡ Auto-refresh every 10s", value=False)
    with col_interval:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()

    stats = api.get_monitoring_stats()

    if stats is None:
        st.error("⚠️ Monitoring service unavailable. Ensure the backend API server is running at localhost:8000.")
        st.stop()

    health = stats.get("health", {})
    traffic = stats.get("traffic", {})
    system = stats.get("system", {})
    drift = stats.get("drift", {})
    perf_drift = stats.get("performance_drift", {})
    history = stats.get("request_history", [])

    # ── Top-line health status ──────────────────────────────────────────────
    api_status = health.get("status", "unknown")
    status_color = "status-pass" if api_status == "healthy" else "status-fail"
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {'#10B981' if api_status=='healthy' else '#EF4444'};">
        <div class="metric-title">API / Model Health</div>
        <div class="metric-value {status_color}">{"🟢 " if api_status == "healthy" else "🔴 "}{api_status.upper()}</div>
        <div style="color: var(--text-secondary); font-size: 12px; margin-top: 6px;">
            Model: {health.get("active_model_version", "N/A")} &nbsp;|&nbsp;
            Model Loaded: {"✓" if health.get("model_loaded") else "✗"} &nbsp;|&nbsp;
            Pipeline Loaded: {"✓" if health.get("pipeline_loaded") else "✗"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Traffic Metrics ─────────────────────────────────────────────────
    st.markdown("### 📊 Request Traffic Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)

    total_req = traffic.get("total_requests", 0)
    avg_lat = traffic.get("avg_latency_ms")
    error_count = traffic.get("error_count", 0)
    error_rate = traffic.get("error_rate_pct", 0.0)
    latest_ts = traffic.get("latest_activity")

    with c1:
        render_metric("Total Prediction Requests", total_req)
    with c2:
        if avg_lat is None:
            render_metric("Avg Latency", "No requests yet", "status-warn")
        else:
            render_metric("Avg Latency", f"{avg_lat:.2f} ms")
    with c3:
        render_metric("Error Count", error_count, "status-fail" if error_count > 0 else "")
    with c4:
        render_metric("Error Rate", f"{error_rate:.1f}%", "status-fail" if error_rate > 1 else "")
    with c5:
        render_metric("Latest Activity", latest_ts[:19].replace("T", " ") if latest_ts else "No requests yet", "")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 1: Request Volume Timeline + System Usage ────────────────
    st.markdown("### 📈 Request Volume & System Usage")
    ch1, ch2 = st.columns(2)

    with ch1:
        # Chart 1: Request volume over time (built from request history)
        if len(history) == 0:
            st.info("📭 No prediction requests received yet. Send requests to `/predict` to populate this chart.")
        else:
            df_hist = pd.DataFrame(history)
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
            df_hist["minute"] = df_hist["timestamp"].dt.floor("T").astype(str)
            df_vol = df_hist.groupby("minute").size().reset_index(name="requests")
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=df_vol["minute"],
                y=df_vol["requests"],
                marker_color="#6366F1",
                name="Requests"
            ))
            fig_vol.add_trace(go.Scatter(
                x=df_vol["minute"],
                y=df_vol["requests"],
                mode="lines+markers",
                line=dict(color="#10B981", width=2),
                name="Trend"
            ))
            fig_vol.update_layout(
                title="Request Volume Over Time",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=320, legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig_vol, use_container_width=True)

    with ch2:
        # Chart 2: System resource usage (CPU + RAM)
        cpu_pct = system.get("cpu_percent", 0.0)
        ram_used = system.get("ram_used_gb", 0.0)
        ram_total = system.get("ram_total_gb", 1.0)
        ram_pct = system.get("ram_usage_percent", 0.0)

        fig_sys = go.Figure()
        fig_sys.add_trace(go.Bar(
            x=["CPU Usage", "RAM Usage"],
            y=[cpu_pct, ram_pct],
            text=[f"{cpu_pct:.1f}%", f"{ram_pct:.1f}%"],
            textposition="outside",
            marker_color=["#F59E0B", "#3B82F6"],
            name="Usage %"
        ))
        fig_sys.update_layout(
            title=f"System Resource Utilisation (RAM: {ram_used:.1f} / {ram_total:.1f} GB)",
            yaxis=dict(range=[0, 105], title="Percentage (%)"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=320,
        )
        st.plotly_chart(fig_sys, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 2: Latency Distribution + Drift Score ───────────────────
    st.markdown("### 🔎 Latency Distribution & Input Drift")
    ch3, ch4 = st.columns(2)

    with ch3:
        # Chart 3: Inference latency distribution
        if len(history) == 0:
            st.info("📭 No prediction requests yet — latency chart unavailable.")
        else:
            latencies = [l["latency_ms"] for l in history if "latency_ms" in l]
            if latencies:
                fig_lat = go.Figure()
                fig_lat.add_trace(go.Histogram(
                    x=latencies,
                    nbinsx=min(20, len(latencies)),
                    marker_color="#10B981",
                    opacity=0.8,
                    name="Latency (ms)"
                ))
                p50 = float(pd.Series(latencies).quantile(0.50))
                p95 = float(pd.Series(latencies).quantile(0.95))
                fig_lat.add_vline(x=p50, line_dash="dash", line_color="#F59E0B",
                                  annotation_text=f"P50={p50:.1f}ms", annotation_font_color="#F59E0B")
                fig_lat.add_vline(x=p95, line_dash="dash", line_color="#EF4444",
                                  annotation_text=f"P95={p95:.1f}ms", annotation_font_color="#EF4444")
                fig_lat.update_layout(
                    title="Inference Latency Distribution",
                    xaxis_title="Latency (ms)", yaxis_title="Count",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white", height=320
                )
                st.plotly_chart(fig_lat, use_container_width=True)
            else:
                st.warning("No latency data available.")

    with ch4:
        # Chart 4: Input drift monitoring gauge
        drift_status = drift.get("status", "no_baseline")
        drift_score = drift.get("drift_score")
        drifted_feats = drift.get("drifted_features", [])

        if drift_status == "no_baseline":
            st.info("📭 **Drift monitoring**: No training baseline loaded. A model must be trained first to establish the reference distribution.")
        elif drift_status == "insufficient_samples":
            count = drift.get("sample_count", 0)
            st.info(f"📭 **Drift monitoring**: Collecting samples ({count}/3 minimum required). Send more prediction requests.")
        elif drift_status == "error":
            st.warning("⚠️ Drift evaluation encountered an error. Check backend logs.")
        else:
            ds_val = drift_score if drift_score is not None else 0.0
            is_drifted = drift.get("is_drifted", False)
            drift_color = "#EF4444" if is_drifted else "#10B981"

            fig_drift = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(ds_val * 100, 1),
                title={"text": "Input Data Drift Score", "font": {"color": "white", "size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white", "tickfont": {"color": "white"}},
                    "bar": {"color": drift_color},
                    "steps": [
                        {"range": [0, 30], "color": "rgba(16,185,129,0.1)"},
                        {"range": [30, 70], "color": "rgba(245,158,11,0.1)"},
                        {"range": [70, 100], "color": "rgba(239,68,68,0.1)"},
                    ],
                    "threshold": {"line": {"color": "#EF4444", "width": 3}, "thickness": 0.8, "value": 30},
                },
                number={"suffix": "%", "font": {"color": "white", "size": 28}},
            ))
            fig_drift.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=320,
            )
            st.plotly_chart(fig_drift, use_container_width=True)

            if is_drifted and drifted_feats:
                st.warning(f"⚠️ **Drift detected** in {len(drifted_feats)} feature(s): `{'`, `'.join(drifted_feats[:6])}`")
            elif not is_drifted:
                st.success(f"✅ No significant drift detected across {drift.get('total_features', '?')} monitored features.")

    # ── Performance Drift (Ground Truth) ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Model Performance Drift")

    perf_status = perf_drift.get("status", "unavailable_no_ground_truth")
    if perf_status == "evaluated":
        pc1, pc2 = st.columns(2)
        with pc1:
            render_metric("Observed Accuracy (labeled)", f"{perf_drift.get('current_accuracy', 0):.4f}", "status-pass")
        with pc2:
            render_metric("Observed F1 (labeled)", f"{perf_drift.get('current_f1', 0):.4f}", "status-pass")
        st.caption(f"Based on {perf_drift.get('sample_count', 0)} labeled predictions.")
    elif perf_status == "unavailable_no_ground_truth":
        st.info("""
        📌 **Performance drift monitoring** requires ground-truth labels to be submitted.
        After making predictions via `/predict`, submit their true labels to `/monitoring/ground_truth`
        to measure model accuracy and F1 on live traffic.
        """)
    else:
        st.warning(f"Performance drift status: `{perf_status}`")

    # ── Prometheus raw output ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📡 Prometheus Telemetry Output")
    raw_metrics = api.get_metrics()
    if raw_metrics:
        with st.expander("View Raw Prometheus Metrics", expanded=False):
            st.code(raw_metrics, language="text")
    else:
        st.warning("Prometheus metrics endpoint unavailable.")

    # Auto-refresh
    if auto_refresh:
        time.sleep(10)
        st.rerun()

