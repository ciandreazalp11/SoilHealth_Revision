import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from io import BytesIO
import joblib
import time
import base64
from PIL import Image
import warnings
import re  # robust column name normalization

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="Machine Learning-Driven Soil Analysis for Sustainable Agriculture System",
    layout="wide",
    page_icon="🌿",
)

# === THEMES (UNCHANGED) ===
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

# === SESSION STATE ===
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
if "location_tag" not in st.session_state:
    st.session_state["location_tag"] = ""

# === APPLY THEME (UNCHANGED) ===
def apply_theme(theme):
    css = f"""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">
    <style>
    /* ===== base app using your theme colors ===== */
    .stApp {{
      font-family: 'Montserrat', sans-serif !important;
      color: {theme['text_color']};
      min-height: 100vh;
      background: {theme['background_main']};  /* honors Classification/Regression themes */
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

    /* ===== 🌊 3D animated pastel water background (aligned to your app) ===== */
    .stApp::before, .stApp::after {{
        content: "";
        position: absolute;
        left: 0; top: 0;
        width: 200%;
        height: 200%;
        /* soft white “wave” pools; stays beautiful over your dark/colored themes */
        background:
          radial-gradient(circle at 50% 50%, rgba(255,255,255,0.40) 0%, transparent 70%),
          radial-gradient(circle at 70% 70%, rgba(255,255,255,0.30) 0%, transparent 70%),
          radial-gradient(circle at 30% 30%, rgba(255,255,255,0.25) 0%, transparent 70%);
        animation: waveMove 15s infinite linear;
        opacity: 0.4;
        z-index: 0;
        pointer-events: none;
    }}
    .stApp::after {{
        animation-delay: -7s;
        opacity: 0.3;
    }}
    @keyframes waveMove {{
        from {{ transform: translateX(0) translateY(0) rotate(0deg); }}
        to   {{ transform: translateX(-25%) translateY(-25%) rotate(360deg); }}
    }}

    /* ===== sidebar layering fix (uses your theme['sidebar_bg']) ===== */
    section[data-testid="stSidebar"] {{
        position: relative !important;
        z-index: 1 !important;
        overflow-y: auto !important;
        background: {theme['sidebar_bg']} !important;
        backdrop-filter: blur(6px);
        height: 100vh !important;
        border-right: 1px solid rgba(255,255,255,0.25);
    }}
    [data-testid="stAppViewContainer"],
    .main {{
        position: relative !important;
        z-index: 2 !important;
    }}
    .stApp::before, .stApp::after {{ z-index: 0 !important; }}

    /* ===== glassy components, color-safe on both themes ===== */
    [data-testid="stJson"] {{
        background: rgba(240, 248, 255, 0.35) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: {theme['primary_color']} !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }}
    [data-testid="stDataFrame"], .stMetric {{
        background: rgba(255, 255, 255, 0.40) !important;
        backdrop-filter: blur(8px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        color: {theme['text_color']} !important;
    }}

    /* ===== headings: same wave float, using your title color ===== */
    h1, h2, h3 {{
        text-shadow: 0 2px 4px rgba(255,255,255,0.6);
        animation: floatTitle 3s ease-in-out infinite;
    }}
    @keyframes floatTitle {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-3px); }}
    }}

    /* ===== buttons & radios pick your current theme’s gradient/text ===== */
    button, .stRadio label:hover {{
        background: {theme['button_gradient']};
        color: {theme['button_text']} !important;
        border-radius: 10px;
        transition: 0.3s;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    # keep your original extra div if you use it elsewhere (harmless)
    st.markdown('<div class="bg-decor"></div>', unsafe_allow_html=True)



apply_theme(st.session_state["current_theme"])

# === SIDEBAR (UNCHANGED LAYOUT) ===
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-header">
          <h2 class="sidebar-title">🌱 Soil Health System</h2>
          <div class="sidebar-sub">ML-Driven Soil Analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
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
            "icon": {
                "color": st.session_state["current_theme"]["menu_icon_color"],
                "font-size": "18px",
            },
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
            },
            "nav-link-selected": {
                "background-color": st.session_state["current_theme"][
                    "nav_link_selected_bg"
                ]
            },
        },
    )
    st.write("---")
    st.markdown(
        f"<div style='font-size:12px;color:{st.session_state['current_theme']['text_color']};opacity:0.85'>Developed for sustainable agriculture</div>",
        unsafe_allow_html=True,
    )
    if st.session_state["last_sidebar_selected"] != selected:
        st.session_state["page_override"] = None
        st.session_state["last_sidebar_selected"] = selected

page = (
    st.session_state["page_override"]
    if st.session_state["page_override"]
    else selected
)

# === COLUMN MAPPING & BASE HELPERS ===
column_mapping = {
    "pH": [
        "pH",
        "ph",
        "soil_pH",
        "soil ph",
        "soilph",
    ],
    "Nitrogen": [
        "Nitrogen",
        "nitrogen",
        "n",
        "nitrogen_level",
        "total_nitrogen",
    ],
    "Phosphorus": [
        "Phosphorus",
        "phosphorus",
        "p",
        "p2o5",
    ],
    "Potassium": [
        "Potassium",
        "potassium",
        "k",
        "k2o",
    ],
    "Moisture": [
        "Moisture",
        "moisture",
        "soil_moisture",
        "moisture_index",
        "moisture content",
        "moisturecontent",
    ],
    "Organic Matter": [
        "Organic Matter",
        "organic matter",
        "organic_matter",
        "organicmatter",
        "om",
        "oc",
        "orgmatter",
        "organic carbon",
        "organic_carbon",
        "organiccarbon",
    ],
    "Latitude": [
        "Latitude",
        "latitude",
        "lat",
    ],
    "Longitude": [
        "Longitude",
        "longitude",
        "lon",
        "lng",
        "longitude_1",
        "longitude_2",
    ],
    "Fertility_Level": [
        "Fertility_Level",
        "fertility_level",
        "fertility class",
        "fertility_class",
        "fertilityclass",
    ],
    "Province": [
        "Province",
        "province",
        "prov",
    ],
}

required_columns = [
    "pH",
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Moisture",
    "Organic Matter",
]


def normalize_col_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    s = re.sub(r"\d+$", "", s)
    return s


def safe_to_numeric_columns(df, cols):
    numeric_found = []
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            numeric_found.append(c)
    return numeric_found


def download_df_button(
    df,
    filename="final_preprocessed_soil_dataset.csv",
    label="⬇️ Download Cleaned & Preprocessed Data",
):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(label=label, data=buf, file_name=filename, mime="text/csv")


def create_fertility_label(df, col="Nitrogen", q=3):
    labels = ["Low", "Moderate", "High"]
    try:
        fert = pd.qcut(df[col], q=q, labels=labels, duplicates="drop")
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
        return (
            "Moderate",
            "orange",
            "⚠️ Some nutrient imbalance. Consider minor adjustments.",
        )
    return ("Poor", "red", "🚫 Deficient or problematic — take corrective action.")


# === CROP PROFILES ===
CROP_PROFILES = {
    "Rice": {
        "pH": (5.5, 6.5),
        "Nitrogen": (0.25, 0.8),
        "Phosphorus": (15, 40),
        "Potassium": (80, 200),
        "Moisture": (40, 80),
        "Organic Matter": (2.0, 6.0),
    },
    "Corn (Maize)": {
        "pH": (5.8, 7.0),
        "Nitrogen": (0.3, 1.2),
        "Phosphorus": (10, 40),
        "Potassium": (100, 250),
        "Moisture": (20, 60),
        "Organic Matter": (1.5, 4.0),
    },
    "Cassava": {
        "pH": (5.0, 7.0),
        "Nitrogen": (0.1, 0.5),
        "Phosphorus": (5, 25),
        "Potassium": (100, 300),
        "Moisture": (20, 60),
        "Organic Matter": (1.0, 3.5),
    },
    "Vegetables (general)": {
        "pH": (6.0, 7.5),
        "Nitrogen": (0.3, 1.5),
        "Phosphorus": (15, 50),
        "Potassium": (120, 300),
        "Moisture": (30, 70),
        "Organic Matter": (2.0, 5.0),
    },
    "Banana": {
        "pH": (5.5, 7.0),
        "Nitrogen": (0.2, 0.8),
        "Phosphorus": (10, 30),
        "Potassium": (200, 500),
        "Moisture": (40, 80),
        "Organic Matter": (2.0, 6.0),
    },
    "Coconut": {
        "pH": (5.5, 7.5),
        "Nitrogen": (0.1, 0.6),
        "Phosphorus": (5, 25),
        "Potassium": (80, 250),
        "Moisture": (30, 70),
        "Organic Matter": (1.0, 4.0),
    },
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


def evaluate_crop_nutrient_gaps(sample: dict, crop_profile: dict):
    issues = []
    for param, (low, high) in crop_profile.items():
        val = sample.get(param, np.nan)
        if pd.isna(val):
            continue
        v = float(val)
        if v < low:
            issues.append(f"{param} too low")
        elif v > high:
            issues.append(f"{param} too high")
    return issues


def build_crop_evaluation_table(sample_series: pd.Series, top_n: int = 6) -> pd.DataFrame:
    sample = sample_series.to_dict()
    rows = []
    for crop, profile in CROP_PROFILES.items():
        score = crop_match_score(sample, profile)
        issues = evaluate_crop_nutrient_gaps(sample, profile)
        if score >= 0.66:
            suitability = "Good"
        elif score >= 0.33:
            suitability = "Moderate"
        else:
            suitability = "Poor"

        if not issues:
            recommendation = "Soil meets most nutrient requirements for this crop."
        else:
            recommendation = (
                "Improve soil for this crop by addressing: " + ", ".join(issues)
            )

        rows.append(
            {
                "Crop": crop,
                "Suitability": suitability,
                "MatchScore": round(score, 3),
                "LimitingFactors": ", ".join(issues) if issues else "None",
                "Recommendation": recommendation,
            }
        )
    if not rows:
        return pd.DataFrame()
    df_eval = pd.DataFrame(rows)
    df_eval = df_eval.sort_values("MatchScore", ascending=False)
    if top_n and top_n > 0:
        df_eval = df_eval.head(top_n)
    return df_eval


def compute_suitability_score(row, features=None):
    if features is None:
        features = [
            "pH",
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Moisture",
            "Organic Matter",
        ]
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
        base_score = float(np.mean(vals)) / (
            np.max(vals) if np.max(vals) != 0 else 1.0
        )

    fi = None
    feat = None
    if st.session_state.get("results"):
        fi = st.session_state["results"].get("feature_importances")
        feat = st.session_state["results"].get("X_columns")
    if fi and feat:
        weights = {f_name: w for f_name, w in zip(feat, fi)}
        weighted = []
        for f, w in weights.items():
            if f in row and f in (df.columns if df is not None else []) and not pd.isna(
                row[f]
            ):
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


def clip_soil_ranges(df: pd.DataFrame) -> pd.DataFrame:
    if "pH" in df.columns:
        df["pH"] = df["pH"].clip(3.5, 9.0)
    if "Moisture" in df.columns:
        df["Moisture"] = df["Moisture"].clip(0, 100)
    if "Organic Matter" in df.columns:
        df["Organic Matter"] = df["Organic Matter"].clip(lower=0)
    for ncol in ["Nitrogen", "Phosphorus", "Potassium"]:
        if ncol in df.columns:
            q_low = df[ncol].quantile(0.01)
            q_high = df[ncol].quantile(0.99)
            if pd.notna(q_low) and pd.notna(q_high) and q_high > q_low:
                df[ncol] = df[ncol].clip(q_low, q_high)
    return df


def run_kmeans_on_df(
    df: pd.DataFrame, features: list, n_clusters: int = 3
):
    sub = df[features].dropna().copy()
    if sub.shape[0] < n_clusters or n_clusters < 1:
        return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    sub["cluster"] = labels
    return sub, model


def upload_and_preprocess_widget():
    st.markdown("### 📂 Upload Soil Data")
    st.markdown(
        "Upload one or more soil analysis files (.csv or .xlsx). The app will attempt "
        "to standardize column names and auto-preprocess numeric columns."
    )
    uploaded_files = st.file_uploader(
        "Select datasets", type=["csv", "xlsx"], accept_multiple_files=True
    )

    if st.session_state["df"] is not None and not uploaded_files:
        st.success(
            f"✅ Loaded preprocessed dataset "
            f"({st.session_state['df'].shape[0]} rows, "
            f"{st.session_state['df'].shape[1]} cols)."
        )
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
                if hasattr(file, "size") and file.size > 8 * 1024 * 1024:
                    st.warning(f"{file.name} is too large! Must be <8MB.")
                    continue
                if not (file.name.endswith(".csv") or file.name.endswith(".xlsx")):
                    st.warning(f"{file.name}: Unsupported extension.")
                    continue

                if file.name.endswith(".csv"):
                    df_file = pd.read_csv(file)
                else:
                    df_file = pd.read_excel(file)

                if df_file.empty:
                    st.warning(f"{file.name} is empty!")
                    continue
                if len(df_file.columns) < 2:
                    st.warning(f"{file.name} has too few columns for analysis.")
                    continue

                # --- robust column standardization (FIRST match wins) ---
                col_norm_map = {}
                for c in df_file.columns:
                    key = normalize_col_name(c)
                    if key not in col_norm_map:
                        col_norm_map[key] = c  # keep first encountered column

                renamed = {}
                for std_col, alt_names in column_mapping.items():
                    candidates = [std_col] + alt_names
                    for cand in candidates:
                        norm_cand = normalize_col_name(cand)
                        if norm_cand in col_norm_map:
                            src_col = col_norm_map[norm_cand]
                            renamed[src_col] = std_col
                            break

                if renamed:
                    df_file.rename(columns=renamed, inplace=True)

                numeric_core_cols = [
                    col
                    for col in required_columns + ["Latitude", "Longitude"]
                    if col in df_file.columns
                ]
                safe_to_numeric_columns(df_file, numeric_core_cols)

                df_file.drop_duplicates(inplace=True)
                cleaned_dfs.append(df_file)

                recognized = [
                    c
                    for c in required_columns
                    + ["Latitude", "Longitude", "Fertility_Level", "Province"]
                    if c in df_file.columns
                ]
                recog_text = (
                    ", ".join(recognized) if recognized else "no core soil features"
                )
                st.success(
                    f"✅ Cleaned {file.name} — recognized: {recog_text} "
                    f"({df_file.shape[0]} rows)"
                )
            except Exception as e:
                st.warning(f"⚠️ Skipped {file.name}: {e}")

        if cleaned_dfs and any(len(x) > 0 for x in cleaned_dfs):
            df = pd.concat(cleaned_dfs, ignore_index=True, sort=False)
            if df.empty:
                st.error("All loaded files are empty after concatenation.")
                return

            safe_to_numeric_columns(df, required_columns + ["Latitude", "Longitude"])

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
                    df[c].fillna(method="ffill", inplace=True)

            df.dropna(how="all", inplace=True)

            df = clip_soil_ranges(df)

            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                st.error(f"Some required columns missing: {missing_required}")
                return

            if "Nitrogen" in df.columns:
                if "Fertility_Level" not in df.columns or df["Fertility_Level"].nunique() < 2:
                    df["Fertility_Level"] = create_fertility_label(
                        df, col="Nitrogen", q=3
                    )

            st.session_state["df"] = df
            st.success("✨ Dataset preprocessed and stored in session.")
            st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
            st.dataframe(df.head())
            download_df_button(df)
            st.markdown("---")
            st.markdown(
                "When you're ready you can go straight to Modeling or Visualization:"
            )
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
            st.error(
                "No valid sheets processed. Check file formats and column headers."
            )


def render_profile(name, asset_filename):
    st.markdown(
        """
    <style>
    .avatar-card {
        display: flex; flex-direction: column; align-items: center; 
        margin-bottom: 22px;
    }
    .avatar-holo {
        width: 170px; height: 170px;
        border-radius: 50%;
        background: conic-gradient(from 180deg at 50% 50%, #00ffd0, #fff700, #ff00d4, #00ffd0 100%);
        padding: 6px;
        box-shadow: 0 0 19px 2px #00ffd088, 0 0 36px 11px #ff00d455;
        position: relative;
        margin-bottom: 18px;
        transition: box-shadow 0.3s ease;
        animation: hologlow 2.9s infinite alternate;
    }
    @keyframes hologlow {
      to {
        box-shadow: 0 0 9px 6px #fff70077, 0 0 80px 11px #0ffbdd44;
      }
    }
    .avatar-holo img {
        width: 100%; height: 100%; object-fit: cover; border-radius: 50%;
        box-shadow: 0 3px 15px #0004;
        background: #fff;
    }
    .avatar-name {
      font-size: 22px; font-weight: 700;
      color: #00ffd0; 
      margin-bottom: 6px; margin-top: -4px;
      letter-spacing: 1px;
    }
    .avatar-role {
      font-size: 14px;
      color: #444; font-style: italic;
      padding-bottom: 2px;
    }
    .bsis-label {
      margin-top: 7px; margin-bottom: 7px;
      padding: 5px 18px; font-size: 16.5px; font-weight: 700;
      color: #fff; background: linear-gradient(to right, #1dd1ff, #ff75db);
      border-radius: 18px; border: none;
      box-shadow: 0 2px 10px #00ffd066;
      text-align: center; display: inline-block; letter-spacing: 1.3px;
      outline: none;
      transition: background 0.2s;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    asset_path = f"assets/{asset_filename}"
    try:
        image = Image.open(asset_path)
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_html = f'<img src="data:image/png;base64,{img_b64}" alt="profile" />'
    except Exception:
        img_html = (
            '<div style="width:170px;height:170px;background:#eee;border-radius:50%;'
            "display:flex;align-items:center;justify-content:center;color:#aaa;"
            '">No Image</div>'
        )

    role_line = ""
    if "andre" in name.lower():
        role_line = "Developer | Machine Learning, Full Stack, Soil Science"
    elif "rica" in name.lower():
        role_line = "Developer | Data Analysis, Visualization, Soil Science"

    st.markdown(
        f"""
    <div class="avatar-card">
        <div class="avatar-holo">{img_html}</div>
        <div class="avatar-name">{name}</div>
        <div class="avatar-role">{role_line}</div>
        <div class="bsis-label">BSIS-4A</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ---------------- MAIN PAGE LOGIC -----------------
if page == "🏠 Home":
    st.title(
        "Machine Learning-Driven Soil Analysis for Sustainable Agriculture System"
    )
    st.markdown(
        "<small style='color:rgba(255,255,255,0.75)'>Capstone Project</small>",
        unsafe_allow_html=True,
    )
    st.write("---")
    upload_and_preprocess_widget()

elif page == "🤖 Modeling":
    st.title("🤖 Modeling — Random Forest")
    st.markdown(
        "Train Random Forest models for Fertility (Regression) or Soil Health (Classification)."
    )
    if st.session_state["df"] is None:
        st.info("Please upload a dataset first in 'Home'.")
    else:
        df = st.session_state["df"].copy()

        st.markdown("#### Model Mode")
        default_checkbox = (
            True if st.session_state.get("task_mode") == "Regression" else False
        )
        chk = st.checkbox(
            "Switch to Regression mode", value=default_checkbox, key="model_mode_checkbox"
        )
        if chk:
            st.session_state["task_mode"] = "Regression"
            st.session_state["current_theme"] = theme_sakura
        else:
            st.session_state["task_mode"] = "Classification"
            st.session_state["current_theme"] = theme_classification

        apply_theme(st.session_state["current_theme"])

        switch_color = (
            "#ff8aa2"
            if st.session_state["task_mode"] == "Regression"
            else "#81c784"
        )
        st.markdown(
            f"""
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
        """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if st.session_state["task_mode"] == "Classification":
            if "Fertility_Level" not in df.columns and "Nitrogen" in df.columns:
                df["Fertility_Level"] = create_fertility_label(df, col="Nitrogen", q=3)
            y = df["Fertility_Level"] if "Fertility_Level" in df.columns else None
        else:
            y = df["Nitrogen"] if "Nitrogen" in df.columns else None

        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        for loccol in ["Latitude", "Longitude"]:
            if loccol in numeric_features:
                numeric_features.remove(loccol)
        if "Nitrogen" in numeric_features:
            numeric_features.remove("Nitrogen")

        st.subheader("Feature Selection")
        st.markdown("Select numeric features to include in the model.")
        selected_features = st.multiselect(
            "Features", options=numeric_features, default=numeric_features
        )

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
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled_df, y, test_size=test_size / 100, random_state=42
            )

            if st.button("🚀 Train Random Forest"):
                if n_estimators > 300:
                    st.info(
                        "High n_estimators may take a while to train! "
                        "Consider lowering for faster results."
                    )
                with st.spinner("Training Random Forest..."):
                    time.sleep(0.25)
                    if st.session_state["task_mode"] == "Classification":
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42,
                            n_jobs=-1,
                        )
                    else:
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42,
                            n_jobs=-1,
                        )

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    try:
                        cv_scores = cross_val_score(
                            model,
                            X_scaled_df,
                            y,
                            cv=5,
                            scoring=(
                                "accuracy"
                                if st.session_state["task_mode"] == "Classification"
                                else "r2"
                            ),
                        )
                        cv_summary = {
                            "mean_cv": float(np.mean(cv_scores)),
                            "std_cv": float(np.std(cv_scores)),
                        }
                    except Exception:
                        cv_summary = None

                    try:
                        perm_imp = permutation_importance(
                            model,
                            X_test,
                            y_test,
                            n_repeats=10,
                            random_state=42,
                            n_jobs=-1,
                        )
                        perm_df = pd.DataFrame(
                            {
                                "feature": selected_features,
                                "importance": perm_imp.importances_mean,
                            }
                        )
                        perm_df = perm_df.sort_values("importance", ascending=False)
                        perm_data = perm_df.to_dict("records")
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
                        "permutation_importance": perm_data,
                    }
                    st.session_state["trained_on_features"] = selected_features
                    st.success(
                        "✅ Training completed. Go to 'Results' to inspect performance and explanations."
                    )

elif page == "📊 Visualization":
    st.title("📊 Data Visualization")
    st.markdown(
        "Explore distributions, correlations, locations, and relationships in your preprocessed data."
    )
    if st.session_state["df"] is None:
        st.info("Please upload data first in 'Home' (Upload Data is integrated there).")
    else:
        df = st.session_state["df"]
        if "Nitrogen" in df.columns and "Fertility_Level" not in df.columns:
            df["Fertility_Level"] = create_fertility_label(
                df, col="Nitrogen", q=3
            )

        st.subheader("Parameter Overview (Levels & Distributions)")
        param_cols = [
            c
            for c in [
                "pH",
                "Nitrogen",
                "Phosphorus",
                "Potassium",
                "Moisture",
                "Organic Matter",
            ]
            if c in df.columns
        ]
        if not param_cols:
            st.warning(
                "No recognized parameter columns found. Required example columns: "
                "pH, Nitrogen, Phosphorus, Potassium, Moisture, Organic Matter"
            )
        else:
            for col in param_cols:
                fig = px.histogram(
                    df,
                    x=col,
                    nbins=30,
                    marginal="box",
                    title=f"Distribution: {col}",
                    color_discrete_sequence=[
                        st.session_state["current_theme"]["primary_color"]
                    ],
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")

        st.subheader("Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            corr = df[numeric_cols].corr()
            corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale=px.colors.sequential.Viridis,
                title="Correlation Heatmap",
            )
            fig_corr.update_traces(
                texttemplate="%{z:.2f}",
                textfont=dict(color="white", size=10),
            )
            fig_corr.update_layout(template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No numeric columns available for correlation matrix.")
        st.markdown("---")

        # === Location map using Latitude/Longitude/Province, auto-focused on data in PH ===
        if "Latitude" in df.columns and "Longitude" in df.columns:
            st.subheader("🗺️ Location map of soil samples (auto-zoom on PH data)")
            loc_df = df.dropna(subset=["Latitude", "Longitude"]).copy()
            if not loc_df.empty:
                # Basic sanity filter: keep only plausible PH lat/lon
                loc_df = loc_df[
                    (loc_df["Latitude"].between(4, 22))
                    & (loc_df["Longitude"].between(116, 127))
                ]

                if loc_df.empty:
                    st.info(
                        "No latitude/longitude values fall inside Philippines bounds (4–22 N, 116–127 E)."
                    )
                else:
                    prov_col = (
                        "Province"
                        if "Province" in loc_df.columns
                        else ("province" if "province" in loc_df.columns else None)
                    )

                    color_col = None
                    if "Fertility_Level" in loc_df.columns:
                        color_col = "Fertility_Level"
                    elif prov_col is not None:
                        color_col = prov_col

                    hover_cols = []
                    if prov_col and prov_col in loc_df.columns:
                        hover_cols.append(prov_col)
                    for col in ["soil_type"]:
                        if col in loc_df.columns and col not in hover_cols:
                            hover_cols.append(col)
                    for col in [
                        "pH",
                        "Nitrogen",
                        "Phosphorus",
                        "Potassium",
                        "Moisture",
                        "Organic Matter",
                    ]:
                        if col in loc_df.columns and col not in hover_cols:
                            hover_cols.append(col)

                    lat_min = float(loc_df["Latitude"].min())
                    lat_max = float(loc_df["Latitude"].max())
                    lon_min = float(loc_df["Longitude"].min())
                    lon_max = float(loc_df["Longitude"].max())
                    lat_center = (lat_min + lat_max) / 2.0
                    lon_center = (lon_min + lon_max) / 2.0

                    if color_col:
                        fig_geo = px.scatter_geo(
                            loc_df,
                            lat="Latitude",
                            lon="Longitude",
                            color=color_col,
                            hover_name=prov_col if prov_col else None,
                            hover_data=hover_cols,
                            title="Soil sample locations by fertility/province (Philippines)",
                        )
                    else:
                        fig_geo = px.scatter_geo(
                            loc_df,
                            lat="Latitude",
                            lon="Longitude",
                            hover_data=hover_cols,
                            title="Soil sample locations (Philippines)",
                        )

                    fig_geo.update_geos(
                        scope="asia",
                        center=dict(lat=lat_center, lon=lon_center),
                        lataxis_range=[lat_min - 0.5, lat_max + 0.5],
                        lonaxis_range=[lon_min - 0.5, lon_max + 0.5],
                        showcoastlines=True,
                        showcountries=True,
                        countrycolor="white",
                        coastlinecolor="white",
                        projection_type="mercator",
                    )
                    fig_geo.update_layout(template="plotly_dark", height=500)
                    st.plotly_chart(fig_geo, use_container_width=True)
            else:
                st.info(
                    "Latitude/Longitude present but all rows are NaN; map cannot be drawn."
                )

elif page == "📈 Results":
    st.title("📈 Model Results & Interpretation")
    if not st.session_state.get("results"):
        st.info(
            "No trained model in session. Train a model first (Modeling or Quick Model)."
        )
    else:
        results = st.session_state["results"]
        task = results["task"]
        y_test = np.array(results["y_test"])
        y_pred = np.array(results["y_pred"])

        st.subheader("Model Summary")
        colA, colB = st.columns([3, 2])
        with colA:
            st.write(f"**Model:** {results.get('model_name', 'Random Forest')}")
            st.write(f"**Features:** {', '.join(results.get('X_columns', []))}")
            if results.get("cv_summary"):
                cv = results["cv_summary"]
                st.write(
                    f"Cross-val mean: **{cv['mean_cv']:.3f}** "
                    f"(std: {cv['std_cv']:.3f})"
                )
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

        metrics_col, explain_col = st.columns([2, 1])
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
                    cm = confusion_matrix(
                        y_test, y_pred, labels=["Low", "Moderate", "High"]
                    )
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title="Confusion Matrix (Low / Moderate / High)",
                    )
                    fig_cm.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig_cm, use_container_width=True)
                except Exception:
                    st.write("Confusion matrix not available")

                st.markdown("#### 📊 Classification Report (Detailed)")
                try:
                    rep = classification_report(
                        y_test, y_pred, output_dict=True
                    )
                    rep_df = pd.DataFrame(rep).transpose().reset_index()
                    rep_df.rename(columns={"index": "Class"}, inplace=True)
                    cols_order = [
                        "Class",
                        "precision",
                        "recall",
                        "f1-score",
                        "support",
                    ]
                    rep_df = rep_df[
                        [c for c in cols_order if c in rep_df.columns]
                    ]
                    st.dataframe(rep_df[cols_order], use_container_width=True)
                except Exception:
                    st.text(classification_report(y_test, y_pred))
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.metric("RMSE", f"{rmse:.3f}")
                st.metric("MAE", f"{mae:.3f}")
                st.metric("R²", f"{r2:.3f}")

                df_res = pd.DataFrame(
                    {"Actual_Nitrogen": y_test, "Predicted_Nitrogen": y_pred}
                )
                st.markdown("**Sample predictions**")
                st.dataframe(df_res.head(10), use_container_width=True)

                st.markdown("**Actual vs Predicted**")
                try:
                    fig1 = px.scatter(
                        df_res,
                        x="Actual_Nitrogen",
                        y="Predicted_Nitrogen",
                        trendline="ols",
                        title="Actual vs Predicted Nitrogen (Model Predictions)",
                    )
                    fig1.update_layout(template="plotly_dark")
                    st.plotly_chart(fig1, use_container_width=True)
                except Exception:
                    fig1 = px.scatter(
                        df_res,
                        x="Actual_Nitrogen",
                        y="Predicted_Nitrogen",
                        title="Actual vs Predicted Nitrogen (no trendline available)",
                    )
                    fig1.update_layout(template="plotly_dark")
                    st.plotly_chart(fig1, use_container_width=True)

                df_res["residual"] = (
                    df_res["Actual_Nitrogen"] - df_res["Predicted_Nitrogen"]
                )
                fig_res = px.histogram(
                    df_res,
                    x="residual",
                    nbins=30,
                    title="Residual Distribution",
                )
                fig_res.update_layout(template="plotly_dark")
                st.plotly_chart(fig_res, use_container_width=True)

        with explain_col:
            st.subheader("What the metrics mean")
            if task == "Classification":
                st.markdown("- **Accuracy:** Overall fraction of correct predictions.")
                st.markdown(
                    "- **Confusion Matrix:** Rows = true classes, Columns = predicted classes."
                )
                st.markdown(
                    "- **Precision:** Of all predicted positive, how many were actually positive."
                )
                st.markdown(
                    "- **Recall:** Of all actual positive samples, how many were found."
                )
                st.markdown(
                    "- **F1-score:** Harmonic mean of precision and recall; balanced measure."
                )
            else:
                st.markdown(
                    "- **RMSE:** Root Mean Squared Error — lower is better; same units as target."
                )
                st.markdown(
                    "- **MAE:** Mean Absolute Error — average magnitude of errors."
                )
                st.markdown(
                    "- **R²:** Proportion of variance explained by the model (1 is perfect)."
                )

        st.markdown("---")

        st.subheader("🌳 Random Forest Feature Importance")
        feat_names = results.get("X_columns", [])
        fi_list = results.get("feature_importances", [])

        if fi_list and feat_names:
            fi_df = pd.DataFrame(
                {"feature": feat_names, "importance": fi_list}
            ).sort_values("importance", ascending=False)
            fig_fi = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Mean Decrease in Impurity (Feature Importance)",
            )
            fig_fi.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_fi, use_container_width=True)
            st.dataframe(fi_df, use_container_width=True)
        else:
            st.info("Feature importances not available for this run.")

        perm_data = results.get("permutation_importance")
        if perm_data:
            st.subheader("🔁 Permutation Importance (robust importance)")
            perm_df = pd.DataFrame(perm_data)
            perm_df = perm_df.sort_values("importance", ascending=False)
            fig_perm = px.bar(
                perm_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Permutation Importance",
            )
            fig_perm.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_perm, use_container_width=True)
            st.dataframe(perm_df, use_container_width=True)
        else:
            st.info("Permutation importance not computed or unavailable.")

        if fi_list and feat_names:
            fi_pairs = list(zip(feat_names, fi_list))
            fi_pairs.sort(key=lambda x: x[1], reverse=True)
            top_feats = [name for name, _ in fi_pairs[:3]]
            st.markdown(
                f"_The model relies most on: **{', '.join(top_feats)}** when making predictions._"
            )

        st.markdown("---")

        st.subheader("🔍 Prediction Explorer — Soil Health from Custom Sample")
        model = st.session_state.get("model")
        scaler = st.session_state.get("scaler")
        df_full = st.session_state.get("df")

        if not model or not scaler:
            st.info(
                "Train a model and keep it in memory to use the prediction explorer."
            )
        elif not feat_names:
            st.info("Feature list is unavailable; retrain the model to populate it.")
        elif df_full is None:
            st.info("Original dataset not found; cannot derive input ranges.")
        else:
            with st.form("prediction_form"):
                st.markdown(
                    "Provide a hypothetical soil sample to see predicted fertility "
                    "or Nitrogen level."
                )
                input_values = {}
                for f in feat_names:
                    if f in df_full.columns and pd.api.types.is_numeric_dtype(
                        df_full[f]
                    ):
                        col_min = float(df_full[f].min())
                        col_max = float(df_full[f].max())
                        if col_min == col_max:
                            col_min -= 0.1
                            col_max += 0.1
                        default_val = float(df_full[f].median())
                        step = (col_max - col_min) / 100.0
                        if step <= 0:
                            step = 0.01
                        input_values[f] = st.slider(
                            f,
                            min_value=float(col_min),
                            max_value=float(col_max),
                            value=float(default_val),
                            step=float(step),
                        )
                    else:
                        input_values[f] = st.number_input(f, value=0.0)

                submitted = st.form_submit_button("Predict Soil Health")
            if submitted:
                sample_df = pd.DataFrame([input_values])
                try:
                    sample_scaled = scaler.transform(sample_df[feat_names])
                    pred = model.predict(sample_scaled)[0]

                    if task == "Classification":
                        pred_label = str(pred)
                        health_label, color, message = interpret_label(pred_label)
                        st.markdown(
                            f"**Predicted Fertility Class:** `{pred_label}`<br>"
                            f"**Soil Health Interpretation:** **{health_label}** — {message}",
                            unsafe_allow_html=True,
                        )
                    else:
                        pred_nitrogen = float(pred)
                        st.markdown(
                            f"**Predicted Nitrogen:** `{pred_nitrogen:.3f}` (same units as your dataset)"
                        )
                        if (
                            df_full is not None
                            and "Nitrogen" in df_full.columns
                            and df_full["Nitrogen"].notna().sum() > 5
                        ):
                            q = df_full["Nitrogen"].quantile([0.33, 0.66])
                            low_th, high_th = q.iloc[0], q.iloc[1]
                            if pred_nitrogen <= low_th:
                                fert = "Low"
                            elif pred_nitrogen <= high_th:
                                fert = "Moderate"
                            else:
                                fert = "High"
                            health_label, color, message = interpret_label(fert)
                            st.markdown(
                                f"**Derived Fertility Category:** `{fert}`<br>"
                                f"**Soil Health Interpretation:** **{health_label}** — {message}",
                                unsafe_allow_html=True,
                            )

                    if fi_list and feat_names:
                        fi_pairs = list(zip(feat_names, fi_list))
                        fi_pairs.sort(key=lambda x: x[1], reverse=True)
                        top_reasons = ", ".join(
                            [name for name, _ in fi_pairs[:3]]
                        )
                        st.markdown(
                            f"_Model explanation:_ This model is globally most sensitive "
                            f"to **{top_reasons}** when predicting soil health."
                        )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

elif page == "🌿 Insights":
    st.title("🌿 Soil Health Insights & Crop Recommendations")
    if st.session_state["df"] is None:
        st.info("Upload and preprocess a dataset first (Home).")
    else:
        df = st.session_state["df"].copy()
        features = [
            "pH",
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Moisture",
            "Organic Matter",
        ]

        st.session_state["location_tag"] = st.text_input(
            "Optional location / farm name (for context in reports)",
            value=st.session_state.get("location_tag", ""),
        )
        context_label = (
            st.session_state["location_tag"].strip()
            if st.session_state["location_tag"]
            else "this dataset"
        )

        if all(f in df.columns for f in features):
            median_row = df[features].median()
            overall_score = compute_suitability_score(median_row, features=features)
            label, color_hex = suitability_color(overall_score)
            crops = recommend_crops_for_sample(median_row, top_n=3)
            crop_list = ", ".join([c[0] for c in crops])

            fertility_map = {
                "Green": (
                    "Good",
                    "🟢",
                    "Fertility Level: Good",
                    "Soil health is optimal for cropping and sustainability.",
                ),
                "Orange": (
                    "Moderate",
                    "🟠",
                    "Fertility Level: Moderate",
                    "Soil is moderately fertile. Soil improvement can boost yield.",
                ),
                "Red": (
                    "Poor",
                    "🔴",
                    "Fertility Level: Poor",
                    "Soil has low fertility; significant amendments are needed.",
                ),
            }
            level_text, circle, level_label, description = fertility_map[label]

            st.markdown(
                f"""
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
                <div style='font-size:18px;font-weight:600;padding-top:2px;'>
                    Soil health summary for <b>{context_label}</b>.
                </div>
                <div style='font-size:20px;font-weight:600;padding-top:6px;'>
                    {description}
                </div>
                <div style='font-size:16px;padding-top:8px;'>
                    <b>Recommended crops (overall suitability):</b> 
                    <span style='color:{color_hex};font-weight:700'>{crop_list}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.subheader("Dataset overview")
        st.write(f"Samples: {df.shape[0]}  — Columns: {df.shape[1]}")
        st.markdown("---")

        st.subheader("Sample-level suitability & agriculture validation")

        display_cols = [
            c
            for c in [
                "Province",
                "pH",
                "Nitrogen",
                "Phosphorus",
                "Potassium",
                "Moisture",
                "Organic Matter",
                "Latitude",
                "Longitude",
                "Fertility_Level",
            ]
            if c in df.columns
        ]

        preview = df[display_cols].copy()
        preview["suitability_score"] = preview.apply(
            lambda r: compute_suitability_score(
                r,
                features=[
                    "pH",
                    "Nitrogen",
                    "Phosphorus",
                    "Potassium",
                    "Moisture",
                    "Organic Matter",
                ],
            ),
            axis=1,
        )
        preview["suitability_label"] = preview["suitability_score"].apply(
            lambda s: suitability_color(s)[0]
        )
        preview["suitability_hex"] = preview["suitability_score"].apply(
            lambda s: suitability_color(s)[1]
        )

        def _rec_small(row):
            s = recommend_crops_for_sample(row, top_n=3)
            return ", ".join([f"{c} ({score:.2f})" for c, score in s])

        preview["top_crops"] = preview.apply(lambda r: _rec_small(r), axis=1)

        def agriculture_verdict(row):
            score = row["suitability_score"]
            label, hex_color = suitability_color(score)
            crops = recommend_crops_for_sample(row, top_n=3)
            crop_names = ", ".join([c[0] for c in crops])
            if label == "Green":
                verdict = (
                    f"🟢 Soil is sustainable for cropping. "
                    f"Ideal crops: {crop_names}."
                )
            elif label == "Orange":
                verdict = (
                    f"🟠 Soil is moderately suitable. "
                    f"Consider minor fertilizer or pH adjustment. "
                    f"Crops that may grow: {crop_names}."
                )
            else:
                verdict = (
                    f"🔴 Soil is NOT recommended for cropping without amendments. "
                    f"Major improvement needed. Possible crops (with treatment): "
                    f"{crop_names}."
                )
            return verdict

        preview["verdict"] = preview.apply(agriculture_verdict, axis=1)

        col_g, col_o, col_r = st.columns(3)
        for label_name, col_obj in zip(
            ["Green", "Orange", "Red"], [col_g, col_o, col_r]
        ):
            cnt = int((preview["suitability_label"] == label_name).sum())
            col_obj.metric(f"{label_name} samples", cnt)

        st.dataframe(
            preview[
                [
                    col
                    for col in [
                        "Province",
                        "pH",
                        "Nitrogen",
                        "Phosphorus",
                        "Potassium",
                        "Moisture",
                        "Organic Matter",
                        "Fertility_Level",
                        "suitability_score",
                        "suitability_label",
                        "top_crops",
                        "verdict",
                    ]
                    if col in preview.columns
                ]
            ],
            use_container_width=True,
        )
        st.markdown("---")

        if "Province" in preview.columns:
            st.subheader("Per-province soil health summary")
            prov_summary = (
                preview.groupby("Province")
                .agg(
                    samples=("suitability_score", "count"),
                    avg_suitability=("suitability_score", "mean"),
                    green_samples=(
                        "suitability_label",
                        lambda x: (x == "Green").sum(),
                    ),
                    orange_samples=(
                        "suitability_label",
                        lambda x: (x == "Orange").sum(),
                    ),
                    red_samples=(
                        "suitability_label",
                        lambda x: (x == "Red").sum(),
                    ),
                )
                .reset_index()
            )
            prov_summary["avg_suitability"] = prov_summary["avg_suitability"].round(3)
            st.dataframe(prov_summary, use_container_width=True)
            st.markdown("---")

        st.markdown("### Soil Suitability Color Legend")
        st.markdown(
            """
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
            <b>Actions:</b> Nutrient/fertilizer adjustment and checking pH.<br>
            <b>Crops:</b> Corn, Cassava, selected vegetables.</td>
          </tr>
          <tr>
            <td><span style="color:#e74c3c;font-weight:900;font-size:20px;">🔴 Red</span></td>
            <td><b>Poor/Unsuitable</b>. Not recommended for cropping.<br>
            <b>Actions:</b> Major improvement with organic matter, fertilizers, and pH correction.<br>
            <b>Crops:</b> Only hardy types after soil amendment.</td>
          </tr>
        </table>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.subheader("Detailed crop evaluation for a specific soil sample")
        if df.shape[0] > 0:
            idx = st.number_input(
                "Select sample index (0-based)",
                min_value=0,
                max_value=int(df.shape[0] - 1),
                value=0,
                step=1,
            )
            sample_row = df.iloc[int(idx)]
            eval_df = build_crop_evaluation_table(sample_row, top_n=6)
            if not eval_df.empty:
                st.dataframe(eval_df, use_container_width=True)
            else:
                st.info(
                    "Unable to compute crop evaluation for this sample (missing values?)."
                )
        else:
            st.info("No samples available for detailed crop evaluation.")

        st.markdown("---")

        st.subheader("Soil pattern clustering (K-Means)")
        cluster_features = [f for f in features if f in df.columns]
        if len(cluster_features) < 2:
            st.info(
                "Need at least two numeric soil parameters (e.g., pH and Nitrogen) "
                "to run clustering."
            )
        else:
            n_clusters = st.slider("Number of clusters", 2, 5, 3, step=1)
            clustered_df, kmeans_model = run_kmeans_on_df(
                df, cluster_features, n_clusters=n_clusters
            )
            if clustered_df is None:
                st.info(
                    "Not enough valid rows to run clustering with the selected number of clusters."
                )
            else:
                counts = clustered_df["cluster"].value_counts().sort_index()
                st.write("Cluster sizes:")
                st.write(counts)

                x_feat = cluster_features[0]
                y_feat = cluster_features[1]
                fig_cluster = px.scatter(
                    clustered_df,
                    x=x_feat,
                    y=y_feat,
                    color="cluster",
                    title=f"K-Means clusters using {x_feat} vs {y_feat}",
                )
                fig_cluster.update_layout(template="plotly_dark")
                st.plotly_chart(fig_cluster, use_container_width=True)

elif page == "👤 About":
    st.title("👤 About the Makers")
    st.markdown("<div style='font-size:19px;'>Developed by:</div>", unsafe_allow_html=True)
    st.write("")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        render_profile("Andre Oneal A. Plaza", "andre_oneal_a._plaza.png")
    with col_b:
        render_profile("Rica Baliling", "rica_baliling.png")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:15px;color:#cd5fff;font-weight:600;'>All glory to God.</div>",
        unsafe_allow_html=True,
    )
    st.write("Developed for a capstone project.")
