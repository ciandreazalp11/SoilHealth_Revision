import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from io import BytesIO
import joblib
import time
import base64
from PIL import Image
import io as sysio
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="Machine Learning-Driven Soil Analysis for Sustainable Agriculture System",
    layout="wide",
    page_icon="🌿"
)

# Themes as in your previous code

theme_classification = {
    "background_main": "linear-gradient(120deg, #0f2c2c 0%, #1a4141 40%, #0e2a2a 100%)",
    "sidebar_bg": "rgba(15, 30, 30, 0.95)",
    "primary_color": "#81c784",
    "secondary_color": "#a5d6a7",
    "button_gradient": "linear-gradient(90deg, #66bb6a, #4caf50)",
    "button_text": "#0c1d1d",
    "header_glow_color_1": "#81c784",
    "header_glow_color_2": "#4caf50",
    "menu_icon_color": "#81c784",
    "nav_link_color": "#e0ffe0",
    "nav_link_selected_bg": "#4caf50",
    "info_bg": "#214242",
    "info_border": "#4caf50",
    "success_bg": "#2e5c2e",
    "success_border": "#81c784",
    "warning_bg": "#5c502e",
    "warning_border": "#dcd380",
    "error_bg": "#5c2e2e",
    "error_border": "#ef9a9a",
    "text_color": "#e0ffe0",
    "title_color": "#a5d6a7",
}

theme_sakura = {
    "background_main": "linear-gradient(120deg, #2b062b 0%, #3b0a3b 50%, #501347 100%)",
    "sidebar_bg": "linear-gradient(180deg, rgba(30,8,30,0.95), rgba(45,10,45,0.95))",
    "primary_color": "#ff8aa2",
    "secondary_color": "#ffc1d3",
    "button_gradient": "linear-gradient(90deg, #ff8aa2, #ff3b70)",
    "button_text": "#1f0f16",
    "header_glow_color_1": "#ff93b0",
    "header_glow_color_2": "#ff3b70",
    "menu_icon_color": "#ff93b0",
    "nav_link_color": "#ffd6e0",
    "nav_link_selected_bg": "#ff3b70",
    "info_bg": "#40132a",
    "info_border": "#ff93b0",
    "success_bg": "#3a1b2a",
    "success_border": "#ff93b0",
    "warning_bg": "#3b2530",
    "warning_border": "#ffb3b3",
    "error_bg": "#3a1a22",
    "error_border": "#ff9aa3",
    "text_color": "#ffeef8",
    "title_color": "#ffd6e0",
}

if "current_theme" not in st.session_state:
    st.session_state["current_theme"] = theme_classification
if "df" not in st.session_state:
    st.session_state["df"] = None
if "results" not in st.session_state:
    st.session_state["results"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None
if "task_mode" not in st.session_state:
    st.session_state["task_mode"] = "Classification"
if "trained_on_features" not in st.session_state:
    st.session_state["trained_on_features"] = None
if "profile_andre" not in st.session_state:
    st.session_state["profile_andre"] = None
if "profile_rica" not in st.session_state:
    st.session_state["profile_rica"] = None
if "page_override" not in st.session_state:
    st.session_state["page_override"] = None
if "last_sidebar_selected" not in st.session_state:
    st.session_state["last_sidebar_selected"] = None

def apply_theme(theme):
    css = f"""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">
    <style>
    .stApp {{
      font-family: 'Montserrat', sans-serif !important;
      color:{theme['text_color']};
      min-height:100vh;
      background: {theme['background_main']};
      background-attachment: fixed;
      position: relative;
      overflow: hidden;
    }}
    html, body, .stApp, .markdown-text-container, [data-testid="stMarkdownContainer"] {{
      font-size: 17px !important;
      line-height: 1.58em !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
      font-family: 'Playfair Display', serif !important;
      color: {theme['title_color']};
      font-weight: 700 !important;
      margin-bottom: 0.7em !important;
    }}
    .block-container {{
      padding-top: 1.5rem !important;
      padding-bottom: 1.5rem !important;
      max-width: 1200px !important;
      margin-left: auto !important;
      margin-right: auto !important;
    }}
    .stButton>button, .stDownloadButton>button {{
      font-size: 17px !important;
      padding: 0.6rem 1.2rem !important;
      border-radius: 10px !important;
    }}
    .stDataFrame, .dataframe, .stTable, .stTable-container {{
      font-size: 16px !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.markdown('<div class="bg-decor"></div>', unsafe_allow_html=True)
apply_theme(st.session_state["current_theme"])

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-header">
          <h2 class="sidebar-title">🌱 Soil Health System</h2>
          <div class="sidebar-sub">ML-Driven Soil Analysis</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("---")
    selected = option_menu(
        None,
        ["🏠 Home", "🤖 Modeling", "📊 Visualization", "📈 Results", "🌿 Insights", "👤 About"],
        icons=["house", "robot", "bar-chart", "graph-up", "lightbulb", "person-circle"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": st.session_state["current_theme"]["menu_icon_color"], "font-size": "18px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": st.session_state["current_theme"]["nav_link_selected_bg"]},
        }
    )
    st.write("---")
    st.markdown(f"<div style='font-size:12px;color:{st.session_state['current_theme']['text_color']};opacity:0.85'>Developed for sustainable agriculture</div>", unsafe_allow_html=True)
    if st.session_state["last_sidebar_selected"] != selected:
        st.session_state["page_override"] = None
        st.session_state["last_sidebar_selected"] = selected

page = st.session_state["page_override"] if st.session_state["page_override"] else selected

column_mapping = {
    'pH': ['pH', 'ph', 'Soil_pH'],
    'Nitrogen': ['Nitrogen', 'N', 'Nitrogen_Level'],
    'Phosphorus': ['Phosphorus', 'P'],
    'Potassium': ['Potassium', 'K'],
    'Moisture': ['Moisture', 'Soil_Moisture'],
    'Organic Matter': ['Organic Matter','OrganicMatter', 'OM', 'oc'],
    'Latitude': ['Latitude', 'Lat', 'lat'],
    'Longitude': ['Longitude', 'Lon', 'Lng', 'lon', 'lng']
}
required_columns = ['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Organic Matter']

def safe_to_numeric_columns(df, cols):
    numeric_found = []
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            numeric_found.append(c)
    return numeric_found

def download_df_button(df, filename="final_preprocessed_soil_dataset.csv", label="⬇️ Download Cleaned & Preprocessed Data"):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(label=label, data=buf, file_name=filename, mime="text/csv")

def create_fertility_label(df, col="Nitrogen", q=3):
    labels = ['Low', 'Moderate', 'High']
    try:
        fert = pd.qcut(df[col], q=q, labels=labels, duplicates='drop')
        if fert.nunique() < 3:
            fert = pd.cut(df[col], bins=3, labels=labels)
    except Exception:
        fert = pd.cut(df[col], bins=3, labels=labels, include_lowest=True)
    return fert.astype(str)

def interpret_label(label):
    l = str(label).lower()
    if l in ["high", "good", "healthy", "3", "2.0"]:
        return ("Good", "green", "✅ Nutrients are balanced. Ideal for most crops.")
    if l in ["moderate", "medium", "2", "1.0"]:
        return ("Moderate", "orange", "⚠️ Some nutrient imbalance. Consider minor adjustments.")
    return ("Poor", "red", "🚫 Deficient or problematic — take corrective action.")

CROP_PROFILES = {
    "Rice": {"pH": (5.5, 6.5), "Nitrogen": (0.25, 0.8), "Phosphorus": (15, 40), "Potassium": (80, 200), "Moisture": (40, 80), "Organic Matter": (2.0, 6.0)},
    "Corn (Maize)": {"pH": (5.8, 7.0), "Nitrogen": (0.3, 1.2), "Phosphorus": (10, 40), "Potassium": (100, 250), "Moisture": (20, 60), "Organic Matter": (1.5, 4.0)},
    "Cassava": {"pH": (5.0, 7.0), "Nitrogen": (0.1, 0.5), "Phosphorus": (5, 25), "Potassium": (100, 300), "Moisture": (20, 60), "Organic Matter": (1.0, 3.5)},
    "Vegetables (general)": {"pH": (6.0, 7.5), "Nitrogen": (0.3, 1.5), "Phosphorus": (15, 50), "Potassium": (120, 300), "Moisture": (30, 70), "Organic Matter": (2.0, 5.0)},
    "Banana": {"pH": (5.5, 7.0), "Nitrogen": (0.2, 0.8), "Phosphorus": (10, 30), "Potassium": (200, 500), "Moisture": (40, 80), "Organic Matter": (2.0, 6.0)},
    "Coconut": {"pH": (5.5, 7.5), "Nitrogen": (0.1, 0.6), "Phosphorus": (5, 25), "Potassium": (80, 250), "Moisture": (30, 70), "Organic Matter": (1.0, 4.0)}
}

def crop_match_score(sample: dict, crop_profile: dict):
    scores = []
    for k, rng in crop_profile.items():
        if k not in sample or pd.isna(sample[k]):
            continue
        val = float(sample[k])
        low, high = rng
        if low <= val <= high:
            scores.append(1.0)
        else:
            width = max(1e-6, high - low)
            if val < low:
                dist = (low - val) / width
            else:
                dist = (val - high) / width
            s = max(0.0, np.exp(-dist))
            scores.append(s)
    if not scores:
        return 0.0
    return float(np.mean(scores))

def recommend_crops_for_sample(sample_series: pd.Series, top_n=3):
    sample = sample_series.to_dict()
    scored = []
    for crop, profile in CROP_PROFILES.items():
        s = crop_match_score(sample, profile)
        scored.append((crop, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

def compute_suitability_score(row, features=None):
    if features is None:
        features = ['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Organic Matter']
    vals = []
    for f in features:
        if f not in row or pd.isna(row[f]):
            continue
        vals.append(row[f])
    if not vals:
        return 0.0
    df = st.session_state.get("df")
    if df is not None:
        score_components = []
        for f in features:
            if f not in df.columns or f not in row or pd.isna(row[f]):
                continue
            low = df[f].quantile(0.05)
            high = df[f].quantile(0.95)
            if high - low <= 0:
                norm = 0.5
            else:
                norm = (row[f] - low) / (high - low)
            norm = float(np.clip(norm, 0, 1))
            score_components.append(norm)
        if not score_components:
            return 0.0
        base_score = float(np.mean(score_components))
    else:
        base_score = float(np.mean(vals)) / (np.max(vals) if np.max(vals) != 0 else 1.0)
    fi = None
    if st.session_state.get("results"):
        fi = st.session_state["results"].get("feature_importances")
        feat = st.session_state["results"].get("X_columns")
    if fi and feat:
        weights = {}
        for f_name, w in zip(feat, fi):
            weights[f_name] = w
        weighted = []
        for f, w in weights.items():
            if f in row and f in df.columns and not pd.isna(row[f]):
                low = df[f].quantile(0.05)
                high = df[f].quantile(0.95)
                if high - low <= 0:
                    norm = 0.5
                else:
                    norm = (row[f] - low) / (high - low)
                norm = float(np.clip(norm, 0, 1))
                weighted.append(norm * w)
        if weighted:
            wsum = float(np.sum(list(weights.values())))
            if wsum > 0:
                return float(np.sum(weighted) / wsum)
    return base_score

def suitability_color(score):
    if score >= 0.66:
        return ("Green", "#2ecc71")
    if score >= 0.33:
        return ("Orange", "#f39c12")
    return ("Red", "#e74c3c")

def upload_and_preprocess_widget():
    st.markdown("### 📂 Upload Soil Data")
    st.markdown("Upload one or more soil analysis files (.csv or .xlsx). The app will attempt to standardize column names and auto-preprocess numeric columns.")
    uploaded_files = st.file_uploader("Select datasets", type=['csv', 'xlsx'], accept_multiple_files=True)
    if st.session_state["df"] is not None and not uploaded_files:
        st.success(f"✅ Loaded preprocessed dataset ({st.session_state['df'].shape[0]} rows, {st.session_state['df'].shape[1]} cols).")
        st.dataframe(st.session_state["df"].head())
        if st.button("🔁 Clear current dataset"):
            st.session_state["df"] = None
            st.session_state["results"] = None
            st.session_state["model"] = None
            st.session_state["scaler"] = None
            st.experimental_rerun()
    cleaned_dfs = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                if hasattr(file, 'size') and file.size > 8*1024*1024:
                    st.warning(f"{file.name} is too large! Must be <8MB.")
                    continue
                if not (file.name.endswith('.csv') or file.name.endswith('.xlsx')):
                    st.warning(f"{file.name}: Unsupported extension.")
                    continue
                df_file = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                if df_file.empty:
                    st.warning(f"{file.name} is empty!")
                    continue
                if len(df_file.columns) < 2:
                    st.warning(f"{file.name} has too few columns for analysis.")
                    continue
                renamed = {}
                for std_col, alt_names in column_mapping.items():
                    for alt in alt_names:
                        if alt in df_file.columns:
                            renamed[alt] = std_col
                            break
                df_file.rename(columns=renamed, inplace=True)
                cols_to_keep = [col for col in required_columns + ['Latitude','Longitude'] if col in df_file.columns]
                df_file = df_file[cols_to_keep]
                safe_to_numeric_columns(df_file, cols_to_keep)
                df_file.drop_duplicates(inplace=True)
                cleaned_dfs.append(df_file)
                st.success(f"✅ Cleaned {file.name} — kept: {', '.join(cols_to_keep)} ({df_file.shape[0]} rows)")
            except Exception as e:
                st.warning(f"⚠️ Skipped {file.name}: {e}")
        if cleaned_dfs and any([len(x) > 0 for x in cleaned_dfs]):
            df = pd.concat(cleaned_dfs, ignore_index=True, sort=False)
            if df.empty:
                st.error("All loaded files are empty after concatenation.")
                return
            safe_to_numeric_columns(df, required_columns + ['Latitude','Longitude'])
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                medians = df[numeric_cols].median()
                df[numeric_cols] = df[numeric_cols].fillna(medians)
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            for c in cat_cols:
                try:
                    if df[c].isnull().sum() > 0:
                        df[c].fillna(df[c].mode().iloc[0], inplace=True)
                except Exception:
                    df[c].fillna(method='ffill', inplace=True)
            df.dropna(how='all', inplace=True)
            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                st.error(f"Some required columns missing: {missing_required}")
                return
            st.session_state["df"] = df
            st.success("✨ Dataset preprocessed and stored in session.")
            st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
            st.dataframe(df.head())
            download_df_button(df)
            st.markdown("---")
            st.markdown("When you're ready you can go straight to Modeling or Visualization:")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("➡️ Proceed to Modeling"):
                    st.session_state["page_override"] = "🤖 Modeling"
                    st.experimental_rerun()
            with col2:
                if st.button("➡️ Proceed to Visualization"):
                    st.session_state["page_override"] = "📊 Visualization"
                    st.experimental_rerun()
        else:
            st.error("No valid sheets processed. Check file formats and column headers.")

if page == "🏠 Home":
    st.title("Machine Learning-Driven Soil Analysis for Sustainable Agriculture System")
    st.markdown("<small style='color:rgba(255,255,255,0.75)'>Capstone Project</small>", unsafe_allow_html=True)
    st.write("---")
    upload_and_preprocess_widget()

elif page == "🤖 Modeling":
    st.title("🤖 Modeling — Random Forest")
    st.markdown("Train Random Forest models for Fertility (Regression) or Soil Health (Classification).")
    if st.session_state["df"] is None:
        st.info("Please upload a dataset first in 'Home'.")
    else:
        df = st.session_state["df"].copy()
        st.markdown("#### Model Mode")
        default_checkbox = True if st.session_state.get("task_mode") == "Regression" else False
        chk = st.checkbox("Switch to Regression mode", value=default_checkbox, key="model_mode_checkbox")
        if chk:
            st.session_state["task_mode"] = "Regression"
            st.session_state["current_theme"] = theme_sakura
        else:
            st.session_state["task_mode"] = "Classification"
            st.session_state["current_theme"] = theme_classification
        apply_theme(st.session_state["current_theme"])
        switch_color = "#ff8aa2" if st.session_state["task_mode"] == "Regression" else "#81c784"
        st.markdown(f"""
        <style>
        .fake-switch {{
            width:70px;
            height:36px;
            border-radius:20px;
            background:{switch_color};
            display:inline-block;
            position:relative;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        }}
        .fake-knob {{
            width:28px;height:28px;border-radius:50%;
            background:rgba(255,255,255,0.95); position:absolute; top:4px;
            transition: all .18s ease;
        }}
        .knob-left {{ left:4px; }}
        .knob-right {{ right:4px; }}
        .switch-label {{ font-weight:600; margin-left:10px; color:{st.session_state['current_theme']['text_color']}; }}
        </style>
        <div style="display:flex;align-items:center;margin-bottom:10px;">
          <div class="fake-switch">
            <div class="fake-knob {'knob-right' if st.session_state['task_mode']=='Regression' else 'knob-left'}"></div>
          </div>
          <div class="switch-label">{'Regression' if st.session_state['task_mode']=='Regression' else 'Classification'}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        if st.session_state["task_mode"] == "Classification":
            if 'Nitrogen' in df.columns:
                df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
            y = df['Fertility_Level'] if 'Fertility_Level' in df.columns else None
        else:
            y = df['Nitrogen'] if 'Nitrogen' in df.columns else None
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        for loccol in ['Latitude', 'Longitude']:
            if loccol in numeric_features:
                numeric_features.remove(loccol)
        if 'Nitrogen' in numeric_features:
            numeric_features.remove('Nitrogen')
        st.subheader("Feature Selection")
        st.markdown("Select numeric features to include in the model.")
        selected_features = st.multiselect("Features", options=numeric_features, default=numeric_features)
        if not selected_features:
            st.warning("Select at least one feature.")
        else:
            X = df[selected_features]
            st.subheader("Hyperparameters")
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("n_estimators", 50, 500, 150, step=50)
            with col2:
                max_depth = st.slider("max_depth", 2, 50, 12)
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)
            test_size = st.slider("Test set fraction (%)", 10, 40, 20, step=5)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=test_size/100, random_state=42)
            if st.button("🚀 Train Random Forest"):
                if n_estimators > 300:
                    st.info("High n_estimators may take a while to train! Consider lowering for faster results.")
                with st.spinner("Training Random Forest..."):
                    time.sleep(0.25)
                    if st.session_state["task_mode"] == "Classification":
                        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                    else:
                        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    try:
                        cv_scores = cross_val_score(model, X_scaled_df, y, cv=5, scoring='accuracy' if st.session_state["task_mode"] == "Classification" else 'r2')
                        cv_summary = {"mean_cv": float(np.mean(cv_scores)), "std_cv": float(np.std(cv_scores))}
                    except Exception:
                        cv_summary = None
                    try:
                        perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
                        perm_df = pd.DataFrame({"feature": selected_features, "importance": perm_imp.importances_mean})
                        perm_df = perm_df.sort_values("importance", ascending=False)
                        perm_data = perm_df.to_dict('records')
                    except Exception:
                        perm_data = None
                    st.session_state["model"] = model
                    st.session_state["scaler"] = scaler
                    st.session_state["results"] = {
                        "task": st.session_state["task_mode"],
                        "y_test": y_test.tolist(),
                        "y_pred": np.array(y_pred).tolist(),
                        "model_name": f"Random Forest {st.session_state['task_mode']} Model",
                        "X_columns": selected_features,
                        "feature_importances": model.feature_importances_.tolist(),
                        "cv_summary": cv_summary,
                        "permutation_importance": perm_data
                    }
                    st.session_state["trained_on_features"] = selected_features
                    st.success("✅ Training completed. Go to 'Results' to inspect performance.")

elif page == "📊 Visualization":
    st.title("📊 Data Visualization")
    st.markdown("Explore distributions, correlations, and relationships in your preprocessed data.")
    if st.session_state["df"] is None:
        st.info("Please upload data first in 'Home' (Upload Data is integrated there).")
    else:
        df = st.session_state["df"]
        if 'Nitrogen' in df.columns and 'Fertility_Level' not in df.columns:
            df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
        st.subheader("Parameter Overview (Levels & Distributions)")
        param_cols = [c for c in ['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Organic Matter'] if c in df.columns]
        if not param_cols:
            st.warning("No recognized parameter columns found. Required example columns: pH, Nitrogen, Phosphorus, Potassium, Moisture, Organic Matter")
        else:
            for col in param_cols:
                fig = px.histogram(df, x=col, nbins=30, marginal="box", title=f"Distribution: {col}", color_discrete_sequence=[st.session_state["current_theme"]["primary_color"]])
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
        st.subheader("Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.sequential.Viridis, title="Correlation Heatmap")
            fig_corr.update_layout(template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No numeric columns available for correlation matrix.")
        st.markdown("---")

elif page == "📈 Results":
    st.title("📈 Model Results & Interpretation")
    if not st.session_state.get("results"):
        st.info("No trained model in session. Train a model first (Modeling or Quick Model).")
    else:
        results = st.session_state["results"]
        task = results["task"]
        y_test = np.array(results["y_test"])
        y_pred = np.array(results["y_pred"])
        st.subheader("Model Summary")
        colA, colB = st.columns([3,2])
        with colA:
            st.write(f"**Model:** {results.get('model_name','Random Forest')}")
            st.write(f"**Features:** {', '.join(results.get('X_columns',[]))}")
            if results.get("cv_summary"):
                cv = results["cv_summary"]
                st.write(f"Cross-val mean: **{cv['mean_cv']:.3f}** (std: {cv['std_cv']:.3f})")
        with colB:
            if st.button("💾 Save Model"):
                if st.session_state.get("model"):
                    joblib.dump(st.session_state["model"], "rf_model.joblib")
                    st.success("Model saved as rf_model.joblib")
                else:
                    st.warning("No model in session to save.")
            if st.button("💾 Save Scaler"):
                if st.session_state.get("scaler"):
                    joblib.dump(st.session_state["scaler"], "scaler.joblib")
                    st.success("Scaler saved as scaler.joblib")
                else:
                    st.warning("No scaler in session to save.")
        st.markdown("---")
        metrics_col, explain_col = st.columns([2,1])
        with metrics_col:
            st.subheader("Performance Metrics")
            if task == "Classification":
                try:
                    acc = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{acc:.3f}")
                except Exception:
                    st.write("Accuracy N/A")
                st.markdown("**Confusion Matrix**")
                try:
                    cm = confusion_matrix(y_test, y_pred, labels=['Low','Moderate','High'])
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=px.colors.sequential.Viridis, title="Confusion Matrix (Low / Moderate / High)")
                    fig_cm.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig_cm, use_container_width=True)
                except Exception:
                    st.write("Confusion matrix not available")
                st.markdown("#### 📊 Classification Report (Detailed)")
                try:
                    rep = classification_report(y_test, y_pred, output_dict=True)
                    rep_df = pd.DataFrame(rep).transpose().reset_index()
                    rep_df.rename(columns={"index":"Class"}, inplace=True)
                    cols_order = ["Class","precision","recall","f1-score","support"]
                    rep_df = rep_df[[c for c in cols_order if c in rep_df.columns]]
                    st.dataframe(rep_df[cols_order], use_container_width=True)
                except Exception as e:
                    st.text(classification_report(y_test,y_pred))
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.metric("RMSE", f"{rmse:.3f}")
                st.metric("MAE", f"{mae:.3f}")
                st.metric("R²", f"{r2:.3f}")
                df_res = pd.DataFrame({"Actual_Nitrogen": y_test, "Predicted_Nitrogen": y_pred})
                st.markdown("**Sample predictions**")
                st.dataframe(df_res.head(10), use_container_width=True)
                st.markdown("**Actual vs Predicted**")
                try:
                    fig1 = px.scatter(df_res, x="Actual_Nitrogen", y="Predicted_Nitrogen", trendline="ols",
                                      title="Actual vs Predicted Nitrogen (Model Predictions)")
                    fig1.update_layout(template="plotly_dark")
                    st.plotly_chart(fig1, use_container_width=True)
                except Exception:
                    fig1 = px.scatter(df_res, x="Actual_Nitrogen", y="Predicted_Nitrogen",
                                      title="Actual vs Predicted Nitrogen (no trendline available)")
                    fig1.update_layout(template="plotly_dark")
                    st.plotly_chart(fig1, use_container_width=True)
                df_res["residual"] = df_res["Actual_Nitrogen"] - df_res["Predicted_Nitrogen"]
                fig_res = px.histogram(df_res, x="residual", nbins=30, title="Residual Distribution")
                fig_res.update_layout(template="plotly_dark")
                st.plotly_chart(fig_res, use_container_width=True)
        with explain_col:
            st.subheader("What the metrics mean")
            if task == "Classification":
                st.markdown("- **Accuracy:** Overall fraction of correct predictions.")
                st.markdown("- **Confusion Matrix:** Rows = true classes, Columns = predicted classes.")
                st.markdown("- **Precision:** Of all predicted positive, how many were actually positive.")
                st.markdown("- **Recall:** Of all actual positive samples, how many were found.")
                st.markdown("- **F1-score:** Harmonic mean of precision and recall; balanced measure.")
            else:
                st.markdown("- **RMSE:** Root Mean Squared Error — lower is better; same units as target.")
                st.markdown("- **MAE:** Mean Absolute Error — average magnitude of errors.")
                st.markdown("- **R²:** Proportion of variance explained by the model (1 is perfect).")

elif page == "🌿 Insights":
    st.title("🌿 Soil Health Insights & Crop Recommendations")
    if st.session_state["df"] is None:
        st.info("Upload and preprocess a dataset first (Home).")
    else:
        df = st.session_state["df"].copy()
        features = ['pH','Nitrogen','Phosphorus','Potassium','Moisture','Organic Matter']
        if all(f in df.columns for f in features):
            median_row = df[features].median()
            overall_score = compute_suitability_score(median_row, features=features)
            label, color_hex = suitability_color(overall_score)
            crops = recommend_crops_for_sample(median_row, top_n=3)
            crop_list = ', '.join([c[0] for c in crops])
            fertility_map = {
                "Green": ("Good", "🟢", "Fertility Level: Good", "Soil health is optimal for cropping and sustainability."),
                "Orange": ("Moderate", "🟠", "Fertility Level: Moderate", "Soil is moderately fertile. Some improvement may help."),
                "Red": ("Poor", "🔴", "Fertility Level: Poor", "Soil has low fertility; major amendments needed."),
            }
            level_text, circle, level_label, description = fertility_map[label]
            st.markdown(f"""
            <div style="
                border:2.5px solid {color_hex};
                border-radius:18px;
                background: linear-gradient(100deg, {color_hex}22 0%, #f4fff4 100%);
                padding:28px 22px 16px 22px;
                margin-bottom:34px;
                box-shadow:0 0px 32px 0px {color_hex}33;
                text-align:left;">
                <h3 style='margin-top:0;margin-bottom:0.2em;'>
                    {circle}
                    <span style="
                        color:{color_hex};
                        font-size:33px;
                        font-weight:bold;
                        vertical-align:middle;">
                        {level_label}
                    </span>
                </h3>
                <div style='font-size:20px;font-weight:600;padding-top:4px;'>
                    {description}
                </div>
                <div style='font-size:16px;padding-top:8px;'>
                    <b>Recommended Crops:</b> <span style='color:{color_hex};font-weight:700'>{crop_list}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.subheader("Dataset overview")
        st.write(f"Samples: {df.shape[0]}  — Columns: {df.shape[1]}")
        st.markdown("---")
        st.subheader("Sample-level suitability & recommendations")
        display_cols = [c for c in ['pH','Nitrogen','Phosphorus','Potassium','Moisture','Organic Matter','Latitude','Longitude'] if c in df.columns]
        preview = df[display_cols].copy()
        preview['suitability_score'] = preview.apply(lambda r: compute_suitability_score(r, features=['pH','Nitrogen','Phosphorus','Potassium','Moisture','Organic Matter']), axis=1)
        preview['suitability_label'] = preview['suitability_score'].apply(lambda s: suitability_color(s)[0])
        preview['suitability_hex'] = preview['suitability_score'].apply(lambda s: suitability_color(s)[1])
        def _rec_small(row):
            s = recommend_crops_for_sample(row, top_n=3)
            return ", ".join([f"{c} ({score:.2f})" for c, score in s])
        preview['top_crops'] = preview.apply(lambda r: _rec_small(r), axis=1)
        def agriculture_verdict(row):
            score = row['suitability_score']
            label, hex_color = suitability_color(score)
            crops = recommend_crops_for_sample(row, top_n=3)
            if label == "Green":
                verdict = f"🟢 Soil is sustainable for cropping. Ideal crops: {', '.join([c[0] for c in crops])}."
            elif label == "Orange":
                verdict = f"🟠 Soil is moderately suitable. Consider minor fertilizer or pH adjustment. Crops that may grow: {', '.join([c[0] for c in crops])}."
            else:
                verdict = f"🔴 Soil is NOT recommended for cropping without amendments. Major improvement needed. Possible crops (with treatment): {', '.join([c[0] for c in crops])}."
            return verdict
        preview['verdict'] = preview.apply(agriculture_verdict, axis=1)
        st.dataframe(preview[['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Organic Matter', 'suitability_score', 'suitability_label', 'top_crops', 'verdict']], use_container_width=True)
        st.markdown("---")
        st.markdown("### Soil Suitability Color Legend")
        st.markdown("""
        <style>
        .legend-table {
            width: 97%;
            margin: 0 auto;
            background: rgba(255,255,255,0.06);
            border-radius: 11px;
            border: 1.4px solid #eee;
            box-shadow: 0 4px 16px #0001;
            font-size:17px;
        }
        .legend-table td {
            padding:10px 16px;
        }
        </style>
        <table class="legend-table">
          <tr>
            <td><span style="color:#2ecc71;font-weight:900;font-size:20px;">🟢 Green</span></td>
            <td><b>Good/Sustainable</b>. Soil is ideal for cropping.<br>
            <b>Recommended crops:</b> Rice, Corn, Cassava, Vegetables, Banana, Coconut.</td>
          </tr>
          <tr>
            <td><span style="color:#f39c12;font-weight:900;font-size:20px;">🟠 Orange</span></td>
            <td><b>Moderate</b>. Soil is OK but may require improvement.<br>
            <b>Actions:</b> Nutrient/fertilizer adjustment.<br>
            <b>Crops:</b> Corn, Cassava, selected vegetables.</td>
          </tr>
          <tr>
            <td><span style="color:#e74c3c;font-weight:900;font-size:20px;">🔴 Red</span></td>
            <td><b>Poor/Unsuitable</b>. Not recommended for cropping.<br>
            <b>Actions:</b> Major improvement needed.<br>
            <b>Crops:</b> Only hardy types after soil amendment.</td>
          </tr>
        </table>
        """, unsafe_allow_html=True)

elif page == "👤 About":
    st.title("👤 About the Makers")
    st.markdown("Developed by:")
    st.write("")  # spacing
    col_a, col_b = st.columns([1,1])
    with col_a:
        render_profile("Andre Oneal A. Plaza", "profile_andre", "upload_andre")
    with col_b:
        render_profile("Rica Baliling", "profile_rica", "upload_rica")
    st.markdown("---")
    st.markdown("All glory to God.")
    st.write("Developed for a capstone project.")
