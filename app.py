import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0a0a0f;
    color: #e8e8f0;
}

section[data-testid="stSidebar"] {
    background-color: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}

section[data-testid="stSidebar"] * {
    color: #e8e8f0 !important;
}

section[data-testid="stSidebar"] .stRadio label {
    color: #e8e8f0 !important;
}

.hero-container {
    background: linear-gradient(135deg, #0f0f1a 0%, #131320 50%, #0f0f1a 100%);
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-container::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -5%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.06) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-tag {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #818cf8;
    padding: 0.3rem 1rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin: 0.5rem 0 1rem 0;
    letter-spacing: -0.02em;
}

.hero-title span {
    background: linear-gradient(90deg, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    color: #9ca3af;
    font-size: 1rem;
    font-weight: 300;
    line-height: 1.7;
    max-width: 600px;
}

.metric-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease;
}

.metric-card:hover {
    border-color: #2e2e4e;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #34d399);
}

.metric-label {
    color: #6b7280;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 0.25rem;
}

.metric-sub {
    color: #34d399;
    font-size: 0.8rem;
    font-weight: 500;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e1e2e, transparent);
}

.info-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem;
    height: 100%;
}

.info-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #818cf8;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.info-card ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.info-card ul li {
    color: #9ca3af;
    font-size: 0.875rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid #1a1a2e;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.info-card ul li::before {
    content: '';
    width: 4px;
    height: 4px;
    background: #34d399;
    border-radius: 50%;
    flex-shrink: 0;
}

.info-card ul li:last-child {
    border-bottom: none;
}

.dataset-table {
    width: 100%;
    border-collapse: collapse;
}

.dataset-table td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #1a1a2e;
    font-size: 0.875rem;
}

.dataset-table td:first-child {
    color: #6b7280;
    font-weight: 500;
    width: 40%;
}

.dataset-table td:last-child {
    color: #e8e8f0;
}

.model-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    color: #818cf8;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.2rem;
}

.best-model-banner {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.1), rgba(99, 102, 241, 0.1));
    border: 1px solid rgba(52, 211, 153, 0.2);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.best-model-banner .crown {
    font-size: 1.5rem;
}

.best-model-banner .text strong {
    color: #34d399;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    display: block;
}

.best-model-banner .text span {
    color: #9ca3af;
    font-size: 0.85rem;
}

.performance-metric-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1a1a2e;
}

.performance-metric-row:last-child {
    border-bottom: none;
}

.performance-metric-row .name {
    color: #9ca3af;
    font-size: 0.875rem;
}

.performance-metric-row .bar-container {
    flex: 1;
    margin: 0 1rem;
    height: 6px;
    background: #1a1a2e;
    border-radius: 3px;
    overflow: hidden;
}

.performance-metric-row .bar {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #34d399);
    border-radius: 3px;
}

.performance-metric-row .value {
    color: #ffffff;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.875rem;
    min-width: 50px;
    text-align: right;
}

.form-section {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.form-section h4 {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #818cf8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #1e1e2e;
}

.result-card-high {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.result-card-low {
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.25);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.result-status-high {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #ef4444;
    margin-bottom: 0.25rem;
}

.result-status-low {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #34d399;
    margin-bottom: 0.25rem;
}

.result-desc {
    color: #9ca3af;
    font-size: 0.8rem;
}

.prob-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.prob-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 0.25rem;
}

.prob-label {
    color: #6b7280;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.recommendation-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}

.recommendation-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1rem;
}

.rec-item {
    display: flex;
    gap: 0.75rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid #1a1a2e;
    align-items: flex-start;
}

.rec-item:last-child {
    border-bottom: none;
}

.rec-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-top: 0.4rem;
    flex-shrink: 0;
}

.rec-dot-high {
    background: #ef4444;
}

.rec-dot-low {
    background: #34d399;
}

.rec-text strong {
    color: #e8e8f0;
    font-size: 0.875rem;
    font-weight: 500;
    display: block;
    margin-bottom: 0.2rem;
}

.rec-text span {
    color: #6b7280;
    font-size: 0.8rem;
}

.stat-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}

.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #ffffff;
}

.stat-label {
    color: #6b7280;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}

.filter-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.about-section {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1rem;
}

.about-section h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #818cf8;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e2e;
}

.about-section p, .about-section li {
    color: #9ca3af;
    font-size: 0.875rem;
    line-height: 1.8;
}

.about-section ul {
    padding-left: 1.25rem;
}

.footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid #1e1e2e;
    margin-top: 3rem;
}

.footer p {
    color: #4b5563;
    font-size: 0.8rem;
    margin: 0;
}

.footer a {
    color: #818cf8;
    text-decoration: none;
}

.footer a:hover {
    color: #34d399;
}

.sidebar-nav-label {
    color: #4b5563 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    margin-bottom: 0.5rem !important;
}

div[data-testid="stMetric"] {
    background: transparent !important;
}

div[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
}

.stButton button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}

.stButton button:hover {
    opacity: 0.9 !important;
}

.stSelectbox label, .stSlider label, .stNumberInput label, .stMultiSelect label {
    color: #9ca3af !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
}

div[data-baseweb="select"] > div {
    background-color: #131320 !important;
    border-color: #1e1e2e !important;
    color: #e8e8f0 !important;
}

.stTextInput input, .stNumberInput input {
    background-color: #131320 !important;
    border-color: #1e1e2e !important;
    color: #e8e8f0 !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #0f0f1a !important;
    border-bottom: 1px solid #1e1e2e !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    color: #6b7280 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    font-size: 0.85rem !important;
}

.stTabs [aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom-color: #818cf8 !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem !important;
}

.stAlert {
    background-color: #0f0f1a !important;
    border-color: #1e1e2e !important;
    color: #e8e8f0 !important;
}

hr {
    border-color: #1e1e2e !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

p, li {
    color: #9ca3af;
}

.stMarkdown p {
    color: #9ca3af;
}

div[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
}

.stSlider [data-testid="stThumbValue"] {
    color: #818cf8 !important;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 2rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.1rem; font-weight: 800; color: #ffffff; letter-spacing: -0.01em;'>
            Churn Predict
        </div>
        <div style='font-size: 0.75rem; color: #4b5563; margin-top: 0.25rem;'>
            ML Prediction System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-nav-label">Navigation</p>', unsafe_allow_html=True)

    page = st.radio(
        label="Navigation",
        options=["Home", "Model Performance", "Make Prediction", "Exploratory Analysis", "About"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style='margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #1e1e2e;'>
        <div style='font-size: 0.7rem; color: #4b5563; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.75rem;'>
            Model Status
        </div>
        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
            <div style='width: 6px; height: 6px; background: #34d399; border-radius: 50%;'></div>
            <span style='font-size: 0.8rem; color: #9ca3af;'>Voting Ensemble Active</span>
        </div>
        <div style='display: flex; align-items: center; gap: 0.5rem;'>
            <div style='width: 6px; height: 6px; background: #34d399; border-radius: 50%;'></div>
            <span style='font-size: 0.8rem; color: #9ca3af;'>15 Models Evaluated</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# HOME PAGE
# ============================================================================
if page == "Home":

    st.markdown("""
    <div class="hero-container">
        <div class="hero-tag">Machine Learning · Banking Analytics</div>
        <div class="hero-title">Bank Customer<br><span>Churn Prediction</span></div>
        <div class="hero-subtitle">
            A production-grade machine learning system that identifies customers at risk of churning.
            Built with 15 models, advanced feature engineering, and ensemble methods.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Models</div>
            <div class="metric-value">15</div>
            <div class="metric-sub">Algorithms evaluated</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best Accuracy</div>
            <div class="metric-value">85.9%</div>
            <div class="metric-sub">Voting Ensemble</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ROC AUC Score</div>
            <div class="metric-value">0.87</div>
            <div class="metric-sub">Area under curve</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Features</div>
            <div class="metric-value">26</div>
            <div class="metric-sub">11 original · 15 engineered</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Capabilities</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>ML Models</h4>
            <ul>
                <li>Logistic Regression</li>
                <li>Decision Tree</li>
                <li>Random Forest</li>
                <li>Gradient Boosting</li>
                <li>XGBoost</li>
                <li>LightGBM</li>
                <li>CatBoost</li>
                <li>Voting Ensemble</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Techniques</h4>
            <ul>
                <li>Advanced Feature Engineering</li>
                <li>SMOTE-Tomek Resampling</li>
                <li>GridSearchCV Tuning</li>
                <li>Stratified Cross Validation</li>
                <li>Ensemble Voting</li>
                <li>Feature Importance Analysis</li>
                <li>ROC AUC Optimization</li>
                <li>Class Imbalance Handling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>Dataset</h4>
            <table class="dataset-table">
                <tr><td>Source</td><td>Kaggle</td></tr>
                <tr><td>Samples</td><td>10,000</td></tr>
                <tr><td>Original Features</td><td>11</td></tr>
                <tr><td>Engineered Features</td><td>15</td></tr>
                <tr><td>Target</td><td>Binary Churn</td></tr>
                <tr><td>Class Imbalance</td><td>4:1 ratio</td></tr>
                <tr><td>Missing Values</td><td>None</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Technology Stack</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>Backend</h4>
            <div style='display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem;'>
                <span class="model-badge">Python 3.9</span>
                <span class="model-badge">Scikit-Learn</span>
                <span class="model-badge">XGBoost</span>
                <span class="model-badge">LightGBM</span>
                <span class="model-badge">CatBoost</span>
                <span class="model-badge">Pandas</span>
                <span class="model-badge">NumPy</span>
                <span class="model-badge">Imbalanced-Learn</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Frontend and Visualization</h4>
            <div style='display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem;'>
                <span class="model-badge">Streamlit</span>
                <span class="model-badge">Matplotlib</span>
                <span class="model-badge">Seaborn</span>
                <span class="model-badge">Custom CSS</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MODEL PERFORMANCE PAGE
# ============================================================================
elif page == "Model Performance":

    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <div style='font-family: Syne, sans-serif; font-size: 2rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;'>
            Model Performance
        </div>
        <div style='color: #6b7280; font-size: 0.9rem; margin-top: 0.4rem;'>
            Comprehensive comparison across all 15 machine learning algorithms
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        results_df = pd.read_csv('model_comparison_results.csv', index_col=0)
        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for col in numeric_cols:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

        results_df_sorted = results_df.sort_values('roc_auc', ascending=False)
        best_model_name = results_df_sorted['roc_auc'].idxmax()
        best_roc_auc = results_df_sorted.loc[best_model_name, 'roc_auc']
        best_accuracy = results_df_sorted.loc[best_model_name, 'accuracy']

        st.markdown(f"""
        <div class="best-model-banner">
            <div class="crown">&#127942;</div>
            <div class="text">
                <strong>Best Model: {best_model_name}</strong>
                <span>Accuracy {best_accuracy:.2%} &nbsp;·&nbsp; ROC AUC {best_roc_auc:.4f} &nbsp;·&nbsp; Selected via cross-validated ROC AUC scoring</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metric bars for best model
        st.markdown('<div class="section-header">Best Model Metrics</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            metrics_display = [
                ("Accuracy", results_df_sorted.loc[best_model_name, 'accuracy']),
                ("Precision", results_df_sorted.loc[best_model_name, 'precision']),
                ("Recall", results_df_sorted.loc[best_model_name, 'recall']),
                ("F1 Score", results_df_sorted.loc[best_model_name, 'f1_score']),
                ("ROC AUC", results_df_sorted.loc[best_model_name, 'roc_auc']),
            ]

            for metric_name, metric_val in metrics_display:
                pct = float(metric_val) * 100
                st.markdown(f"""
                <div class="performance-metric-row">
                    <span class="name">{metric_name}</span>
                    <div class="bar-container">
                        <div class="bar" style="width: {pct}%;"></div>
                    </div>
                    <span class="value">{metric_val:.4f}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # ROC AUC chart
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#0f0f1a')

            sorted_data = results_df_sorted['roc_auc'].sort_values()
            colors = ['#6366f1' if name == best_model_name else '#1e1e3e' for name in sorted_data.index]

            bars = ax.barh(range(len(sorted_data)), sorted_data.values, color=colors, height=0.65)

            for i, (val, bar) in enumerate(zip(sorted_data.values, bars)):
                ax.text(val + 0.003, i, f'{val:.3f}', va='center', color='#9ca3af', fontsize=8)

            ax.set_yticks(range(len(sorted_data)))
            ax.set_yticklabels(sorted_data.index, color='#9ca3af', fontsize=8.5)
            ax.set_xlabel('ROC AUC Score', color='#6b7280', fontsize=9)
            ax.tick_params(colors='#6b7280')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#1e1e2e')
            ax.spines['left'].set_color('#1e1e2e')
            ax.set_xlim(0, 1.05)
            ax.grid(axis='x', color='#1e1e2e', linewidth=0.5)
            ax.set_title('ROC AUC by Model', color='#ffffff', fontsize=11, fontweight='bold', pad=12)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown('<div class="section-header">Full Comparison Table</div>', unsafe_allow_html=True)

        display_df = results_df_sorted[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].copy()
        display_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

        st.dataframe(display_df, use_container_width=True)

        st.markdown('<div class="section-header">Visualization</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Model Evaluation Charts", "ROC Curves"])

        with tab1:
            try:
                st.image('model_evaluation_comprehensive.png', use_column_width=True)
            except Exception:
                st.info("Model evaluation image not found.")

        with tab2:
            try:
                st.image('roc_curves_top_models.png', use_column_width=True)
            except Exception:
                st.info("ROC curves image not found.")

    except FileNotFoundError:
        st.error("model_comparison_results.csv not found. Please run the training script first.")
    except Exception as e:
        st.error(f"Error loading model performance data: {str(e)}")


# ============================================================================
# PREDICTION PAGE
# ============================================================================
elif page == "Make Prediction":

    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <div style='font-family: Syne, sans-serif; font-size: 2rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;'>
            Churn Prediction
        </div>
        <div style='color: #6b7280; font-size: 0.9rem; margin-top: 0.4rem;'>
            Enter customer details to generate a real-time churn risk assessment
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="form-section"><h4>Customer Profile</h4>', unsafe_allow_html=True)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", options=["Male", "Female"])
            age = st.slider("Age", min_value=18, max_value=100, value=40)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="form-section"><h4>Account Details</h4>', unsafe_allow_html=True)
            tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)
            balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0)
            num_products = st.selectbox("Number of Products", options=[1, 2, 3, 4])
            estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=200000.0, value=100000.0, step=1000.0)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="form-section"><h4>Banking Behavior</h4>', unsafe_allow_html=True)
            has_cr_card = st.selectbox("Has Credit Card", options=["Yes", "No"])
            is_active = st.selectbox("Is Active Member", options=["Yes", "No"])
            st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("Generate Prediction", type="primary", use_container_width=True)

    if submitted:
        # Feature engineering
        input_data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': estimated_salary
        }

        input_data['AgeGroup'] = int(pd.cut([age], bins=[0, 30, 40, 50, 60, 100], labels=[1, 2, 3, 4, 5])[0])
        input_data['BalanceGroup'] = int(pd.cut([balance], bins=[-0.01, 0.01, 50000, 100000, 150000, 300000], labels=[0, 1, 2, 3, 4])[0])
        input_data['CreditScoreCategory'] = int(pd.cut([credit_score], bins=[0, 580, 670, 740, 800, 900], labels=[1, 2, 3, 4, 5])[0])
        input_data['TenureGroup'] = int(pd.cut([tenure], bins=[-0.01, 2, 5, 10, 15], labels=[1, 2, 3, 4])[0])
        input_data['BalanceToSalaryRatio'] = balance / estimated_salary if estimated_salary > 0 else 0
        input_data['ProductsPerTenure'] = num_products / tenure if tenure > 0 else num_products
        input_data['IsSenior'] = 1 if age > 60 else 0
        input_data['HasZeroBalance'] = 1 if balance == 0 else 0
        input_data['ActiveWithCard'] = input_data['IsActiveMember'] * input_data['HasCrCard']
        input_data['EngagementScore'] = (input_data['IsActiveMember'] * 0.4 + input_data['HasCrCard'] * 0.2 + (num_products / 4) * 0.4)
        input_data['CustomerValueScore'] = (balance / 300000) * 0.5 + (estimated_salary / 200000) * 0.5
        input_data['ActivityScore'] = input_data['IsActiveMember'] * 0.5 + (tenure / 10) * 0.3 + (num_products / 4) * 0.2
        input_data['CreditToAgeRatio'] = credit_score / age
        input_data['IsHighValueCustomer'] = 1 if (balance > 127644 and estimated_salary > 149388) else 0
        input_data['HasMultipleProducts'] = 1 if num_products > 1 else 0

        geography_map = {'France': 0, 'Germany': 1, 'Spain': 2}
        gender_map = {'Female': 0, 'Male': 1}
        input_data['Geography'] = geography_map[geography]
        input_data['Gender'] = gender_map[gender]

        expected_columns = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'AgeGroup', 'BalanceGroup', 'CreditScoreCategory', 'TenureGroup',
            'BalanceToSalaryRatio', 'ProductsPerTenure', 'IsSenior',
            'HasZeroBalance', 'ActiveWithCard', 'EngagementScore',
            'CustomerValueScore', 'ActivityScore', 'CreditToAgeRatio',
            'IsHighValueCustomer', 'HasMultipleProducts'
        ]

        input_df = pd.DataFrame([input_data])
        input_df = input_df[expected_columns]

        try:
            model = pickle.load(open('best_model.pkl', 'rb'))
            scaler = pickle.load(open('scaler.pkl', 'rb'))

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

            st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div class="result-card-high">
                        <div class="result-status-high">High Churn Risk</div>
                        <div class="result-desc">This customer is likely to leave</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-card-low">
                        <div class="result-status-low">Low Churn Risk</div>
                        <div class="result-desc">This customer is likely to stay</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="prob-card">
                    <div class="prob-label">Churn Probability</div>
                    <div class="prob-value" style="color: {'#ef4444' if probability[1] > 0.5 else '#34d399'};">
                        {probability[1]*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="prob-card">
                    <div class="prob-label">Retention Probability</div>
                    <div class="prob-value" style="color: #34d399;">
                        {probability[0]*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Risk bar chart
            st.markdown('<div style="margin-top: 1.5rem;">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 1.5))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#0f0f1a')

            ax.barh([0], [probability[1]], color='#ef4444', alpha=0.85, height=0.5)
            ax.barh([0], [probability[0]], left=[probability[1]], color='#34d399', alpha=0.85, height=0.5)

            ax.text(probability[1] / 2, 0, f'Churn {probability[1]*100:.1f}%',
                   ha='center', va='center', color='white', fontsize=10, fontweight='bold')
            ax.text(probability[1] + probability[0] / 2, 0, f'Retain {probability[0]*100:.1f}%',
                   ha='center', va='center', color='white', fontsize=10, fontweight='bold')

            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            plt.tight_layout(pad=0.5)
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

            # Recommendations
            if prediction == 1:
                st.markdown("""
                <div class="recommendation-card">
                    <h4>Recommended Actions</h4>
                    <div class="rec-item">
                        <div class="rec-dot rec-dot-high"></div>
                        <div class="rec-text">
                            <strong>Immediate Outreach</strong>
                            <span>Schedule a personal consultation with an account manager within 48 hours</span>
                        </div>
                    </div>
                    <div class="rec-item">
                        <div class="rec-dot rec-dot-high"></div>
                        <div class="rec-text">
                            <strong>Retention Offer</strong>
                            <span>Present a customized product bundle with preferential rates tailored to this customer profile</span>
                        </div>
                    </div>
                    <div class="rec-item">
                        <div class="rec-dot rec-dot-high"></div>
                        <div class="rec-text">
                            <strong>Loyalty Program Enrollment</strong>
                            <span>Enroll in premium rewards program with quarterly benefits review</span>
                        </div>
                    </div>
                    <div class="rec-item">
                        <div class="rec-dot rec-dot-high"></div>
                        <div class="rec-text">
                            <strong>Satisfaction Survey</strong>
                            <span>Send a personalized satisfaction survey and follow up on any reported issues within 24 hours</span>
                        </div>
                    </div>
                    <div class="rec-item">
                        <div class="rec-dot rec-dot-high"></div>
                        <div class="rec-text">
                            <strong>Priority Monitoring</strong>
                            <span>Add to high priority watch list for monthly relationship health reviews</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="recommendation-card">
                    <h4>Recommended Actions</h4>
                    <div class="rec-item">
                        <div class="rec-dot rec-dot-low"></div>
                        <div class="rec-text">
                            <strong>Maintain Service Standards</strong>
                            <span>Continue current level of engagement and service quality</span>
                        </div>
                    </div>
                    <div class="rec-item">
                        <div class="rec-dot rec-dot-low"></div>
                        <div class="rec-text">
                            <strong>Cross-Sell Opportunity</strong>
                            <span>Evaluate additional product offerings that align with this customer's financial profile</span>
                        </div>
                    </div>
                    <div class="rec-item">
                        <div class="rec-dot rec-dot-low"></div>
                        <div class="rec-text">
                            <strong>Loyalty Reinforcement</strong>
                            <span>Ensure enrollment in rewards programs and communicate ongoing benefits</span>
                        </div>
                    </div>
                    <div class="rec-item">
                        <div class="rec-dot rec-dot-low"></div>
                        <div class="rec-text">
                            <strong>Annual Review</strong>
                            <span>Schedule routine satisfaction survey to maintain long-term relationship health</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except FileNotFoundError:
            st.error("Model files not found. Ensure best_model.pkl and scaler.pkl are present in the application directory.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")


# ============================================================================
# EXPLORATORY ANALYSIS PAGE
# ============================================================================
elif page == "Exploratory Analysis":

    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <div style='font-family: Syne, sans-serif; font-size: 2rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;'>
            Exploratory Analysis
        </div>
        <div style='color: #6b7280; font-size: 0.9rem; margin-top: 0.4rem;'>
            Interactive exploration of the customer churn dataset
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = pd.read_csv('churn.csv')

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(df):,}</div>
                <div class="stat-label">Total Customers</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="color: #ef4444;">{df['Exited'].sum():,}</div>
                <div class="stat-label">Churned</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            retained = len(df) - df['Exited'].sum()
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="color: #34d399;">{retained:,}</div>
                <div class="stat-label">Retained</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            churn_rate = df['Exited'].mean() * 100
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{churn_rate:.1f}%</div>
                <div class="stat-label">Churn Rate</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_geography = st.multiselect(
                "Geography",
                options=df['Geography'].unique().tolist(),
                default=df['Geography'].unique().tolist()
            )

        with col2:
            age_range = st.slider(
                "Age Range",
                min_value=int(df['Age'].min()),
                max_value=int(df['Age'].max()),
                value=(int(df['Age'].min()), int(df['Age'].max()))
            )

        with col3:
            gender_filter = st.multiselect(
                "Gender",
                options=df['Gender'].unique().tolist(),
                default=df['Gender'].unique().tolist()
            )

        filtered_df = df[
            (df['Geography'].isin(selected_geography)) &
            (df['Age'].between(age_range[0], age_range[1])) &
            (df['Gender'].isin(gender_filter))
        ]

        st.markdown(f"""
        <div style='background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1.5rem; font-size: 0.85rem; color: #818cf8;'>
            Showing {len(filtered_df):,} customers ({len(filtered_df)/len(df)*100:.1f}% of dataset)
        </div>
        """, unsafe_allow_html=True)

        plt.rcParams.update({
            'figure.facecolor': '#0f0f1a',
            'axes.facecolor': '#0f0f1a',
            'axes.edgecolor': '#1e1e2e',
            'axes.labelcolor': '#6b7280',
            'xtick.color': '#6b7280',
            'ytick.color': '#6b7280',
            'text.color': '#9ca3af',
            'grid.color': '#1e1e2e',
            'grid.linewidth': 0.5,
        })

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.5rem; font-weight: 500;">Age Distribution by Churn Status</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#0f0f1a')

            filtered_df[filtered_df['Exited'] == 0]['Age'].hist(bins=25, alpha=0.7, label='Retained', ax=ax, color='#34d399')
            filtered_df[filtered_df['Exited'] == 1]['Age'].hist(bins=25, alpha=0.7, label='Churned', ax=ax, color='#ef4444')

            ax.set_xlabel('Age', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.legend(fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown('<div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.5rem; font-weight: 500;">Churn Rate by Geography</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#0f0f1a')

            churn_by_geo = filtered_df.groupby('Geography')['Exited'].mean() * 100
            bars = ax.bar(churn_by_geo.index, churn_by_geo.values, color='#6366f1', alpha=0.8, width=0.5)

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f'{h:.1f}%',
                       ha='center', va='bottom', fontsize=9, color='#9ca3af')

            ax.set_ylabel('Churn Rate (%)', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.5rem; font-weight: 500;">Churn Rate by Number of Products</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#0f0f1a')

            churn_by_products = filtered_df.groupby('NumOfProducts')['Exited'].mean() * 100
            bars = ax.bar(churn_by_products.index.astype(str), churn_by_products.values, color='#8b5cf6', alpha=0.8, width=0.5)

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f'{h:.1f}%',
                       ha='center', va='bottom', fontsize=9, color='#9ca3af')

            ax.set_xlabel('Number of Products', fontsize=9)
            ax.set_ylabel('Churn Rate (%)', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col4:
            st.markdown('<div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.5rem; font-weight: 500;">Balance Distribution by Churn Status</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#0f0f1a')

            retained_bal = filtered_df[filtered_df['Exited'] == 0]['Balance']
            churned_bal = filtered_df[filtered_df['Exited'] == 1]['Balance']

            bp = ax.boxplot([retained_bal, churned_bal],
                           patch_artist=True,
                           medianprops=dict(color='#ffffff', linewidth=2),
                           whiskerprops=dict(color='#6b7280'),
                           capprops=dict(color='#6b7280'),
                           flierprops=dict(marker='o', markerfacecolor='#6b7280', markersize=3, alpha=0.3))

            bp['boxes'][0].set_facecolor('#34d399')
            bp['boxes'][0].set_alpha(0.4)
            bp['boxes'][1].set_facecolor('#ef4444')
            bp['boxes'][1].set_alpha(0.4)

            ax.set_xticklabels(['Retained', 'Churned'], fontsize=9)
            ax.set_ylabel('Balance ($)', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown('<div class="section-header">Dataset Sample</div>', unsafe_allow_html=True)
        st.dataframe(filtered_df.head(20), use_container_width=True)

    except FileNotFoundError:
        st.error("churn.csv not found. Please ensure the dataset is in the application directory.")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")


# ============================================================================
# ABOUT PAGE
# ============================================================================
elif page == "About":

    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <div style='font-family: Syne, sans-serif; font-size: 2rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;'>
            About
        </div>
        <div style='color: #6b7280; font-size: 0.9rem; margin-top: 0.4rem;'>
            Project methodology, architecture, and technical details
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="about-section">
            <h3>Project Description</h3>
            <p>
                This customer churn prediction system is a comprehensive machine learning application
                designed to identify banking customers at risk of leaving. The system leverages
                advanced ensemble algorithms, feature engineering, and class imbalance techniques
                to deliver accurate, actionable predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-section">
            <h3>Original Features</h3>
            <ul>
                <li>Credit Score</li>
                <li>Geography</li>
                <li>Gender</li>
                <li>Age</li>
                <li>Tenure</li>
                <li>Account Balance</li>
                <li>Number of Products</li>
                <li>Has Credit Card</li>
                <li>Is Active Member</li>
                <li>Estimated Salary</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-section">
            <h3>Evaluation Metrics</h3>
            <ul>
                <li>Accuracy and Precision</li>
                <li>Recall and F1 Score</li>
                <li>ROC AUC Analysis</li>
                <li>Confusion Matrix</li>
                <li>Matthews Correlation Coefficient</li>
                <li>Cohen's Kappa Score</li>
                <li>Precision Recall Curves</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-section">
            <h3>Methodology</h3>
            <ul>
                <li>Comprehensive exploratory data analysis</li>
                <li>Feature engineering with 15 new derived features</li>
                <li>Class imbalance handling using SMOTE-Tomek</li>
                <li>Feature scaling with StandardScaler</li>
                <li>Hyperparameter optimization using GridSearchCV</li>
                <li>Stratified cross validation for robust estimates</li>
                <li>Ensemble voting for improved predictions</li>
                <li>ROC AUC as primary selection criterion</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-section">
            <h3>Engineered Features</h3>
            <ul>
                <li>Age Group</li>
                <li>Balance Group</li>
                <li>Credit Score Category</li>
                <li>Tenure Group</li>
                <li>Balance to Salary Ratio</li>
                <li>Products per Tenure</li>
                <li>Is Senior</li>
                <li>Has Zero Balance</li>
                <li>Active with Card</li>
                <li>Engagement Score</li>
                <li>Customer Value Score</li>
                <li>Activity Score</li>
                <li>Credit to Age Ratio</li>
                <li>Is High Value Customer</li>
                <li>Has Multiple Products</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-section">
            <h3>Future Enhancements</h3>
            <ul>
                <li>Real-time model retraining pipeline</li>
                <li>SHAP values for prediction explainability</li>
                <li>A/B testing framework for model versions</li>
                <li>CRM system integration</li>
                <li>Automated monitoring and alerting</li>
                <li>Batch prediction API endpoint</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    <p>
        Bank Customer Churn Prediction System &nbsp;·&nbsp;
        <a href="https://github.com/manjot7/churn-predict" target="_blank">GitHub Repository</a>
        &nbsp;·&nbsp;
        <a href="https://linkedin.com/in/manjot7" target="_blank">LinkedIn</a>
        &nbsp;·&nbsp;
        Built with Python and Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
