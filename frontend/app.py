import streamlit as st
import pandas as pd
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from api_client import APIClient

st.set_page_config(page_title="MLSecOps Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E2E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2A2A3C;
        margin-bottom: 20px;
        text-align: center;
    }
    .metric-title {
        color: #A0A0B0;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 5px;
        text-transform: uppercase;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 28px;
        font-weight: 700;
    }
    .status-pass { color: #4CAF50; font-weight: bold; }
    .status-fail { color: #F44336; font-weight: bold; }
    .status-warn { color: #FF9800; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize API Client
api = APIClient()

def render_metric(title, value, color_class=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🛡️ MLSecOps")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Dashboard", 
    "New Experiment", 
    "Experiments", 
    "Models", 
    "Monitoring", 
    "Reports"
])

# Use session state to navigate directly to an experiment if selected
if "selected_experiment_id" not in st.session_state:
    st.session_state["selected_experiment_id"] = "latest"

if st.sidebar.button("Check API Connection"):
    health = api.get_health()
    if health:
        st.sidebar.success(f"Connected! API Status: {health.get('status')}")
    else:
        st.sidebar.error("Failed to connect to backend.")

# --- Page: Dashboard ---
if page == "Dashboard":
    st.title("System Dashboard")
    
    exps = api.get_experiments()
    if not exps:
        st.info("No experiments found in the system. Go to 'New Experiment' to start one.")
    else:
        total = len(exps)
        success = sum(1 for e in exps if e.get("status") == "COMPLETED")
        failed = sum(1 for e in exps if e.get("status") == "FAILED")
        sec_alerts = sum(1 for e in exps if e.get("security_score", 1.0) < 1.0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: render_metric("Total Experiments", total)
        with col2: render_metric("Successful Runs", success, "status-pass")
        with col3: render_metric("Failed Runs", failed, "status-fail")
        with col4: render_metric("Security Alerts", sec_alerts, "status-warn" if sec_alerts > 0 else "status-pass")
        
        st.subheader("Recent Experiments")
        df_exps = pd.DataFrame(exps)
        st.dataframe(df_exps.head(10), use_container_width=True)

# --- Page: New Experiment ---
elif page == "New Experiment":
    st.title("New Experiment")
    
    st.markdown("### 1. Dataset Configuration")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("**Dataset Preview**")
        st.dataframe(df.head(5), use_container_width=True)
        st.markdown(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        target_col = st.selectbox("Select Target Column", df.columns.tolist(), index=len(df.columns)-1)
        opt_metric = st.selectbox("Optimization Metric", ["f1", "roc_auc", "accuracy", "precision", "recall"])
        
        models_to_train = st.multiselect("Models to Train", ["logistic_regression", "random_forest", "gradient_boosting"], default=["logistic_regression", "random_forest", "gradient_boosting"])
        
        if st.button("🚀 Run Secure Pipeline", type="primary"):
            with st.spinner("Uploading dataset and preparing environment..."):
                uploaded_file.seek(0)
                res = api.upload_dataset(uploaded_file.name, uploaded_file.getvalue())
                
                if res and res.get("status") == "success":
                    st.success("Dataset uploaded successfully.")
                    
                    run_res = api.run_pipeline(
                        target_column=target_col,
                        opt_metric=opt_metric,
                        models_to_train=models_to_train
                    )
                    
                    if run_res and "experiment_id" in run_res:
                        st.session_state["polling_exp_id"] = run_res["experiment_id"]
                        st.success(f"Pipeline triggered: {run_res['experiment_id']}")
                        st.rerun()
                    else:
                        st.error("Failed to start pipeline.")
                else:
                    st.error("Failed to upload dataset.")
                    
    if "polling_exp_id" in st.session_state:
        exp_id = st.session_state["polling_exp_id"]
        st.markdown(f"**Tracking Experiment:** `{exp_id}`")
        
        # Pipeline Stages visualization
        stages = ["Validation", "Preprocessing", "Training", "Evaluation", "Security", "Model Validation", "Deployment"]
        cols = st.columns(len(stages))
        
        status_box = st.empty()
        
        while True:
            exp_data = api.get_experiment(exp_id)
            if not exp_data:
                status_box.error("Failed to fetch status. Retrying...")
                time.sleep(2)
                continue
                
            status = exp_data.get("status", "UNKNOWN")
            
            # Simple UI update for stages based on status
            for i, stage in enumerate(stages):
                with cols[i]:
                    if status == "QUEUED":
                        st.markdown(f"⏳ {stage}")
                    elif status == "RUNNING":
                        st.markdown(f"🔄 {stage}")
                    elif status == "COMPLETED":
                        st.markdown(f"✅ {stage}")
                    elif status == "FAILED":
                        st.markdown(f"❌ {stage}")
                        
            status_box.info(f"**Status:** {status}")
            
            if status in ["COMPLETED", "FAILED"]:
                st.session_state["selected_experiment_id"] = exp_id
                if status == "COMPLETED":
                    st.success("Experiment completed successfully! Navigate to 'Experiments' or 'Reports' to view.")
                else:
                    st.error(f"Experiment failed: {exp_data.get('error', 'Unknown error')}")
                del st.session_state["polling_exp_id"]
                break
                
            time.sleep(2)

# --- Page: Experiments (Details) ---
elif page == "Experiments":
    st.title("Experiment Details")
    
    exps = api.get_experiments()
    if not exps:
        st.info("No experiments available.")
    else:
        exp_options = ["latest"] + [e["experiment_id"] for e in exps if "experiment_id" in e]
        selected = st.selectbox("Select Experiment", exp_options, 
                                index=exp_options.index(st.session_state["selected_experiment_id"]) if st.session_state["selected_experiment_id"] in exp_options else 0)
        
        st.session_state["selected_experiment_id"] = selected
        exp = api.get_experiment(selected)
        
        if not exp:
            st.error(f"Could not load data for experiment {selected}")
        elif exp.get("status") in ["QUEUED", "RUNNING"]:
            st.info(f"Experiment {selected} is currently {exp.get('status')}...")
        elif exp.get("status") == "FAILED":
            st.error(f"Experiment {selected} Failed.")
            st.code(exp.get("error_message", "Unknown error"))
        else:
            # Main experiment view tabs
            t_over, t_perf, t_sec, t_res, t_pipe = st.tabs(["Overview", "Performance", "Security", "Resources", "Pipeline"])
            
            with t_over:
                st.subheader(f"Experiment: {exp.get('experiment_id')}")
                c1, c2, c3 = st.columns(3)
                ds = exp.get("dataset", {})
                bm = exp.get("best_model", {})
                c1.metric("Dataset", ds.get("name", "N/A"), f"{ds.get('n_samples', 0)} samples")
                c2.metric("Best Model", bm.get("algorithm", "N/A"))
                c3.metric("Status", exp.get("status", "N/A"))
                
                st.markdown("### Deployment & Delivery")
                st.markdown("Ensure security gates have passed before deploying.")
                c4, c5 = st.columns([1, 1])
                with c4:
                    if st.button("Load Model for /predict"):
                        load_res = api.load_model(selected)
                        if load_res and load_res.get("status") == "success":
                            st.success("Model successfully loaded into inference engine.")
                        else:
                            st.error("Failed to load model.")
                with c5:
                    download_url = api.get_model_download_url(selected)
                    st.markdown(f'<a href="{download_url}" download><button style="padding: 0.5rem 1rem; border-radius: 0.5rem; border: none; background-color: #4CAF50; color: white; cursor: pointer;">Download Model Artifacts</button></a>', unsafe_allow_html=True)

            with t_perf:
                rm = exp.get("research_metrics", {})
                st.subheader("Final Performance")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("F1 Score", f"{rm.get('f1_score', 0):.4f}")
                c2.metric("Precision", f"{rm.get('precision', 0):.4f}")
                c3.metric("Recall", f"{rm.get('recall', 0):.4f}")
                c4.metric("Accuracy", f"{rm.get('accuracy', 0):.4f}")
                
                models = exp.get("models", {})
                if models:
                    st.markdown("### Model Comparison")
                    comp_data = []
                    for algo, data in models.items():
                        m = data.get("metrics", {})
                        comp_data.append({
                            "Algorithm": algo,
                            "F1": m.get("f1", 0),
                            "ROC-AUC": m.get("roc_auc", 0),
                            "Latency (ms)": m.get("latency_p95_ms", 0)
                        })
                    df_comp = pd.DataFrame(comp_data)
                    fig = px.bar(df_comp, x="Algorithm", y="F1", color="F1", title="Algorithm F1 Scores")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### Confusion Matrix (Best Model)")
                    cm = bm.get("metrics", {}).get("confusion_matrix")
                    if cm:
                        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
                        st.plotly_chart(fig_cm)

            with t_sec:
                st.subheader("Security Assessment")
                st.metric("Overall Security Score", f"{sec.get('overall_score', 0)*100:.1f}%")
                if sec.get("overall_passed"):
                    st.success("Model PASSED all security gates.")
                else:
                    st.error("Model FAILED one or more security gates. Deployment BLOCKED.")
                
                st.markdown("### Gate Results")
                for g in sec.get("gate_results", []):
                    status = "✅ PASS" if g.get("passed") else "❌ FAIL"
                    with st.expander(f"{status} - {g.get('gate')} (Score: {g.get('score', 0)})"):
                        st.json(g.get("details", {}))
                        
            with t_res:
                res = exp.get("resources", {})
                st.subheader("Resource Usage")
                c1, c2, c3 = st.columns(3)
                c1.metric("Pipeline Duration", f"{res.get('pipeline_duration_sec', 0):.2f}s")
                c2.metric("CPU Count", res.get("cpu_count", 0))
                c3.metric("RAM Available", f"{res.get('ram_available_gb', 0):.1f} GB")
                
            with t_pipe:
                st.subheader("Pipeline Execution Summary")
                st.json(exp.get("data_quality", {}))

# --- Page: Models ---
elif page == "Models":
    st.title("Registered Models")
    
    exps = api.get_experiments()
    models_list = []
    for e in exps:
        exp = api.get_experiment(e.get("experiment_id"))
        if exp and exp.get("best_model"):
            dep = exp.get("deployment", {})
            models_list.append({
                "Experiment": exp.get("experiment_id"),
                "Algorithm": exp.get("best_model", {}).get("algorithm"),
                "F1 Score": exp.get("research_metrics", {}).get("f1_score", 0),
                "Stage": dep.get("stage", "None"),
                "Version": dep.get("model_version", "None")
            })
            
    if models_list:
        st.dataframe(pd.DataFrame(models_list), use_container_width=True)
    else:
        st.info("No registered models found.")

# --- Page: Monitoring ---
elif page == "Monitoring":
    st.title("Runtime Monitoring")
    st.markdown("Live operational metrics from the FastAPI backend.")
    
    stats = api.get_monitoring_stats()
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Inference Requests", stats.get("inference_requests", 0))
        c2.metric("Anomalies Detected", stats.get("anomalies_detected", 0))
        c3.metric("Avg Latency", f"{stats.get('avg_latency_ms', 0):.2f} ms")
        
        st.markdown("### Raw Prometheus Metrics")
        metrics = api.get_metrics()
        if metrics:
            st.code(metrics, language="text")
    else:
        st.error("Failed to fetch monitoring stats from backend.")

# --- Page: Reports ---
elif page == "Reports":
    st.title("Reports & Downloads")
    
    exps = api.get_experiments()
    if not exps:
        st.info("No experiments to report.")
    else:
        exp_id = st.selectbox("Select Experiment", [e["experiment_id"] for e in exps if "experiment_id" in e])
        exp_data = api.get_experiment(exp_id)
        
        if exp_data:
            st.json(exp_data, expanded=False)
            
            json_str = json.dumps(exp_data, indent=2)
            st.download_button(
                label="Download JSON Report",
                data=json_str,
                file_name=f"{exp_id}_report.json",
                mime="application/json",
                type="primary"
            )
