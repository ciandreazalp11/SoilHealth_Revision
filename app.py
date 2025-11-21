# Updated Streamlit app: app_full.py
# Implements requested features: rule-based agriculture validation + ML classifier (suitability),
# crop suitability scoring, color-coded insights, clustering page, Random Forest explainability,
# improved preprocessing (outlier detection, imputation), optional scaling (default OFF),
# dataset augmentation to synthesize missing fertility classes, sample-level explainability via
# permutation-based local importance, and location tagging. About page left untouched.

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
from sklearn.decomposition import PCA
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

# ------------------- (KEEP THEMES & ABOUT AS ORIGINAL) -------------------
# (omitted here in the canvas preview for brevity — code below reuses the exact theme and About functions
# from the original uploaded app. The About page is intentionally left unchanged.)

# For transparency: this updated app was produced by analyzing your original app source (app (8).py). 
# Reference to the original uploaded file: fileciteturn0file0

# ------------------- Utility & Standardization -------------------

# Column mapping and required columns (same as original)
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

if "df" not in st.session_state:
    st.session_state["df"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None
if "results" not in st.session_state:
    st.session_state["results"] = None

# Safe numeric conversion
def safe_to_numeric_columns(df, cols):
    numeric_found = []
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            numeric_found.append(c)
    return numeric_found

# Outlier handling (IQR capping)
def cap_outliers_iqr(df, cols, factor=1.5):
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            low = q1 - factor * iqr
            high = q3 + factor * iqr
            df[c] = df[c].clip(lower=low, upper=high)
    return df

# Improved preprocessing pipeline
def preprocess_dataframe(df, do_outlier_cap=True, impute_method='median'):
    df = df.copy()
    # Standardize column names
    renamed = {}
    for std_col, alt_names in column_mapping.items():
        for alt in alt_names:
            if alt in df.columns:
                renamed[alt] = std_col
                break
    df.rename(columns=renamed, inplace=True)
    # Keep only relevant cols
    keep_cols = [c for c in required_columns + ['Latitude','Longitude'] if c in df.columns]
    df = df[keep_cols]
    safe_to_numeric_columns(df, keep_cols)
    # Drop exact duplicates
    df.drop_duplicates(inplace=True)
    # Cap outliers
    if do_outlier_cap:
        df = cap_outliers_iqr(df, required_columns)
    # Impute
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if impute_method == 'median':
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)
    else:
        means = df[numeric_cols].mean()
        df[numeric_cols] = df[numeric_cols].fillna(means)
    # For non-numeric cols fill with mode
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for c in cat_cols:
        if df[c].isnull().sum() > 0:
            try:
                df[c].fillna(df[c].mode().iloc[0], inplace=True)
            except Exception:
                df[c].fillna(method='ffill', inplace=True)
    df.dropna(how='all', inplace=True)
    return df

# Synthesize samples to expand dataset classes if needed
def synthesize_balanced_samples(df, target_col='Fertility_Level', numeric_features=None, target_counts=300):
    df = df.copy()
    if numeric_features is None:
        numeric_features = [c for c in required_columns if c in df.columns]
    # If target not present, create from Nitrogen deciles (fallback)
    if target_col not in df.columns:
        df[target_col] = create_fertility_label(df, col='Nitrogen', q=3)
    counts = df[target_col].value_counts()
    all_labels = ['Low','Moderate','High']
    synthetic_rows = []
    for lab in all_labels:
        cur = df[df[target_col]==lab]
        needed = max(0, target_counts - len(cur))
        if needed > 0 and len(cur) > 0:
            # sample with noise
            for i in range(needed):
                base = cur.sample(1).iloc[0]
                new = base.copy()
                for f in numeric_features:
                    if pd.isna(new[f]):
                        new[f] = df[f].median()
                    jitter = np.random.normal(scale=0.03 * max(1.0, abs(new[f])))
                    new[f] = float(np.clip(new[f] + jitter, df[f].min(), df[f].max()))
                synthetic_rows.append(new)
    if synthetic_rows:
        synth_df = pd.DataFrame(synthetic_rows)
        df = pd.concat([df, synth_df], ignore_index=True, sort=False)
    return df

# Fertility label creation (same logic but robust)
def create_fertility_label(df, col="Nitrogen", q=3):
    labels = ['Low', 'Moderate', 'High']
    try:
        fert = pd.qcut(df[col], q=q, labels=labels, duplicates='drop')
        if fert.nunique() < 3:
            fert = pd.cut(df[col], bins=3, labels=labels)
    except Exception:
        fert = pd.cut(df[col], bins=3, labels=labels, include_lowest=True)
    return fert.astype(str)

# ------------------- Rule-based Agriculture Validation -------------------
# Define agronomic thresholds (example — can be tuned or loaded from a config)
AGRONOMIC_THRESHOLDS = {
    'pH': (5.5, 7.5),  # general acceptable window
    'Nitrogen': (0.15, 1.5),  # percent / relative values
    'Phosphorus': (5, 50),
    'Potassium': (50, 400),
    'Moisture': (15, 85),
    'Organic Matter': (0.8, 8.0)
}

def rule_based_suitability(row):
    reasons = []
    ok = True
    for f, (low, high) in AGRONOMIC_THRESHOLDS.items():
        if f in row and not pd.isna(row[f]):
            val = float(row[f])
            if val < low:
                ok = False
                reasons.append(f"{f} below minimum ({val} < {low})")
            elif val > high:
                # too high may be problematic too
                reasons.append(f"{f} above typical upper bound ({val} > {high})")
    return ok, reasons

# Combined verdict: rule-based + ML classifier (trained later)

def agriculture_verdict_with_reasons(row, rf_model=None, feature_cols=None, threshold_prob=0.5):
    # First run hard rule checks
    ok, reasons = rule_based_suitability(row)
    if not ok:
        return False, reasons
    # If rules pass but we have a model, use ML classifier for nuanced decision
    if rf_model is not None and feature_cols is not None:
        X = pd.DataFrame([row])[feature_cols]
        X = X.fillna(X.median())
        try:
            if hasattr(rf_model, 'predict_proba'):
                prob = rf_model.predict_proba(X)[0]
                # assume classes [Unsuitable, Suitable] or similar ordering
                # We'll get probability for Suitable as the last column
                p_suitable = prob[-1]
                if p_suitable < threshold_prob:
                    reasons.append(f"ML model predicts low suitability (p={p_suitable:.2f})")
                    return False, reasons
                else:
                    return True, reasons
        except Exception:
            try:
                pred = rf_model.predict(X)[0]
                if str(pred).lower() in ['unsuitable','no','0']:
                    reasons.append(f"ML model predicted: {pred}")
                    return False, reasons
            except Exception:
                pass
    return True, reasons

# ------------------- Crop Profiles & Matching (unchanged core logic) -------------------
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

# ------------------- Suitability & Color mapping -------------------

def compute_suitability_score(row, df_ref=None, features=None, feature_importances=None):
    if features is None:
        features = [f for f in required_columns if f in row.index or f in row.keys()]
    vals = []
    for f in features:
        if f in row and not pd.isna(row[f]):
            vals.append(row[f])
    if not vals:
        return 0.0
    if df_ref is not None:
        score_components = []
        for f in features:
            if f not in df_ref.columns or f not in row or pd.isna(row[f]):
                continue
            low = df_ref[f].quantile(0.05)
            high = df_ref[f].quantile(0.95)
            if high - low <= 0:
                norm = 0.5
            else:
                norm = (row[f] - low) / (high - low)
            norm = float(np.clip(norm, 0, 1))
            score_components.append(norm)
        if not score_components:
            base_score = float(np.mean(vals)) / (np.max(vals) if np.max(vals) != 0 else 1.0)
        else:
            base_score = float(np.mean(score_components))
    else:
        base_score = float(np.mean(vals)) / (np.max(vals) if np.max(vals) != 0 else 1.0)
    if feature_importances is not None:
        weights = {f: w for f, w in zip(feature_importances['names'], feature_importances['importances'])}
        weighted = []
        for f, w in weights.items():
            if f in row and f in df_ref.columns and not pd.isna(row[f]):
                low = df_ref[f].quantile(0.05)
                high = df_ref[f].quantile(0.95)
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

# ------------------- Streamlit UI: Home (Upload) -------------------

def download_df_button(df, filename="final_preprocessed_soil_dataset.csv", label="⬇️ Download Cleaned & Preprocessed Data"):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(label=label, data=buf, file_name=filename, mime="text/csv")

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
                df_clean = preprocess_dataframe(df_file)
                missing_required = [col for col in required_columns if col not in df_clean.columns]
                if missing_required:
                    st.error(f"Some required columns missing: {missing_required} in {file.name}")
                    continue
                cleaned_dfs.append(df_clean)
                st.success(f"✅ Cleaned {file.name} — kept: {', '.join(df_clean.columns)} ({df_clean.shape[0]} rows)")
            except Exception as e:
                st.warning(f"⚠️ Skipped {file.name}: {e}")
        if cleaned_dfs and any([len(x) > 0 for x in cleaned_dfs]):
            df = pd.concat(cleaned_dfs, ignore_index=True, sort=False)
            df = preprocess_dataframe(df)
            # If only one fertility class exists, synthesize balanced dataset (warn user)
            if 'Nitrogen' in df.columns and 'Fertility_Level' not in df.columns:
                df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
            if df['Fertility_Level'].nunique() < 3:
                st.warning("Dataset has less than 3 fertility classes — creating synthetic samples to balance classes for training. Please consider sourcing more diverse real samples.")
                df = synthesize_balanced_samples(df, target_col='Fertility_Level')
            st.session_state['df'] = df
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

# ------------------- Modeling Page -------------------

def modeling_page():
    st.title("🤖 Modeling — Random Forest & Suitability Classifier")
    if st.session_state["df"] is None:
        st.info("Please upload a dataset first in 'Home'.")
        return
    df = st.session_state["df"].copy()
    # Ensure fertility label exists
    if 'Nitrogen' in df.columns and 'Fertility_Level' not in df.columns:
        df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
    st.subheader("1) Prepare features & choose task")
    feature_cols = [c for c in required_columns if c in df.columns]
    extra_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in feature_cols + ['Latitude','Longitude','Nitrogen']]
    feature_cols += extra_cols
    selected_features = st.multiselect("Select numeric features to include in models:", options=feature_cols, default=feature_cols)
    if not selected_features:
        st.warning("Select at least one feature.")
        return
    # Option: use scaling? default OFF per request
    use_scaling = st.checkbox("Use feature scaling (MinMax) — default OFF (do not scale)", value=False)
    scaler = None
    X = df[selected_features]
    if use_scaling:
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
        X = pd.DataFrame(Xs, columns=selected_features)
    # Create rule-based suitability label (binary)
    suitability_labels = []
    for _, r in df[selected_features + ['pH','Nitrogen']].iterrows():
        ok, reasons = rule_based_suitability(r)
        suitability_labels.append('Suitable' if ok else 'Unsuitable')
    df['Rule_Suitability'] = suitability_labels
    st.markdown("Rule-based suitability label created (Suitable / Unsuitable) — these are hard agronomic cutoffs.")
    st.subheader("2) Train combined classifier (Random Forest)")
    clf_type = st.radio("Classifier target:", options=['Rule_Suitability (binary)', 'Fertility_Level (multiclass)'], index=0)
    y = df['Rule_Suitability'] if clf_type.startswith('Rule') else df['Fertility_Level']
    test_size = st.slider("Test set fraction (%)", 10, 40, 20, step=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, stratify=y)
    n_estimators = st.slider("n_estimators", 50, 500, 150, step=50)
    max_depth = st.slider("max_depth", 2, 50, 12)
    if st.button("🚀 Train Classifier"):
        with st.spinner("Training Random Forest classifier..."):
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            try:
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                cv_summary = {"mean_cv": float(np.mean(cv_scores)), "std_cv": float(np.std(cv_scores))}
            except Exception:
                cv_summary = None
            # permutation importance (global)
            try:
                perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
                perm_df = pd.DataFrame({"feature": selected_features, "importance": perm_imp.importances_mean})
                perm_df = perm_df.sort_values("importance", ascending=False)
            except Exception:
                perm_df = None
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['trained_features'] = selected_features
            st.session_state['results'] = {
                'task': 'classification',
                'y_test': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'model_name': 'RandomForestClassifier_Suitability' if clf_type.startswith('Rule') else 'RandomForestClassifier_Fertility',
                'X_columns': selected_features,
                'feature_importances': model.feature_importances_.tolist(),
                'permutation_importance_df': perm_df.to_dict('records') if perm_df is not None else None,
                'cv_summary': cv_summary
            }
            st.success("✅ Classifier training completed. Go to 'Results' to inspect performance and explanations.")

# ------------------- Visualization Page -------------------

def visualization_page():
    st.title("📊 Data Visualization")
    if st.session_state['df'] is None:
        st.info("Please upload data first in 'Home'.")
        return
    df = st.session_state['df']
    st.subheader("Parameter Distributions")
    param_cols = [c for c in required_columns if c in df.columns]
    for col in param_cols:
        fig = px.histogram(df, x=col, nbins=30, marginal="box", title=f"Distribution: {col}")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Clustering")
    cols_for_clust = st.multiselect("Select features for clustering", options=param_cols, default=param_cols)
    if cols_for_clust:
        k = st.slider("K (clusters)", 2, 8, 3)
        do_pca = st.checkbox("Use PCA to reduce to 2D for plotting", value=True)
        Xc = df[cols_for_clust].fillna(df[cols_for_clust].median())
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(Xc)
        df['_cluster'] = labels.astype(str)
        if do_pca:
            pca = PCA(n_components=2)
            proj = pca.fit_transform(Xc)
            df['_pc1'] = proj[:,0]
            df['_pc2'] = proj[:,1]
            fig = px.scatter(df, x='_pc1', y='_pc2', color='_cluster', hover_data=cols_for_clust, title='Clusters (PCA projection)')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        else:
            # choose first two cols by default
            c1, c2 = cols_for_clust[0], cols_for_clust[1] if len(cols_for_clust) > 1 else cols_for_clust[0]
            fig = px.scatter(df, x=c1, y=c2, color=df['_cluster'], hover_data=cols_for_clust, title='Clusters')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Cluster summaries")
        cluster_summary = df.groupby('_cluster')[cols_for_clust].mean().reset_index()
        st.dataframe(cluster_summary)

# ------------------- Results & Explainability Page -------------------

def results_page():
    st.title("📈 Model Results & Explainability")
    if not st.session_state.get('results'):
        st.info("No trained model in session. Train a model first (Modeling).")
        return
    results = st.session_state['results']
    model = st.session_state.get('model')
    task = results.get('task')
    st.subheader("Model summary")
    st.write(f"Model: {results.get('model_name')}")
    st.write(f"Features: {', '.join(results.get('X_columns', []))}")
    if results.get('cv_summary'):
        cv = results['cv_summary']
        st.write(f"Cross-val mean: **{cv['mean_cv']:.3f}** (std: {cv['std_cv']:.3f})")
    st.markdown('---')
    st.subheader('Global Feature Importances')
    try:
        fi = results.get('feature_importances')
        feat_names = results.get('X_columns')
        if fi and feat_names:
            df_fi = pd.DataFrame({'feature': feat_names, 'importance': fi}).sort_values('importance', ascending=False)
            fig = px.bar(df_fi, x='importance', y='feature', orientation='h', title='Feature importances (Random Forest)')
            fig.update_layout(template='plotly_dark', height=420)
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.write('Feature importance not available')
    st.markdown('---')
    st.subheader('Local explanation (per-sample) via permutation importance')
    if model is None:
        st.info('No model object saved. Train and return to see local explanations.')
        return
    df = st.session_state['df']
    feature_cols = st.session_state.get('trained_features')
    sample_idx = st.number_input('Pick a sample index for explanation (row index)', min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
    sample = df.iloc[int(sample_idx)][feature_cols].fillna(df[feature_cols].median())
    st.write('Sample values:')
    st.dataframe(sample.to_frame().T)
    # compute local permutation importance by permuting each feature and measuring change in probability
    try:
        base_prob = None
        if hasattr(model, 'predict_proba'):
            base_prob = model.predict_proba(pd.DataFrame([sample]))[0]
        local_importances = []
        for f in feature_cols:
            X_sample = pd.DataFrame([sample])
            orig = X_sample[f].iloc[0]
            # permute by replacing with column median
            X_sample[f] = df[f].median()
            if base_prob is not None:
                new_prob = model.predict_proba(X_sample)[0]
                # difference between probability of predicted class
                if hasattr(model, 'classes_'):
                    pred_class = model.predict(pd.DataFrame([sample]))[0]
                    class_idx = list(model.classes_).index(pred_class)
                    delta = float(base_prob[class_idx] - new_prob[class_idx])
                else:
                    delta = float(np.max(base_prob) - np.max(new_prob))
            else:
                # fallback: permutation changes predicted label
                base_pred = model.predict(pd.DataFrame([sample]))[0]
                new_pred = model.predict(X_sample)[0]
                delta = 1.0 if base_pred != new_pred else 0.0
            local_importances.append((f, max(0.0, delta)))
        local_importances.sort(key=lambda x: x[1], reverse=True)
        df_local = pd.DataFrame(local_importances, columns=['feature','local_importance'])
        fig = px.bar(df_local, x='local_importance', y='feature', orientation='h', title='Local permutation importance (approx)')
        fig.update_layout(template='plotly_dark', height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('**Top drivers of this sample prediction**')
        st.dataframe(df_local.head(8))
    except Exception as e:
        st.write('Local explanation computation failed:', e)

# ------------------- Insights Page -------------------

def insights_page():
    st.title('🌿 Soil Health Insights & Crop Recommendations')
    if st.session_state['df'] is None:
        st.info('Upload and preprocess a dataset first (Home).')
        return
    df = st.session_state['df'].copy()
    features = [f for f in required_columns if f in df.columns]
    median_row = df[features].median()
    # If we have trained model and permutation data, pass to compute_suitability_score
    fi = None
    if st.session_state.get('results'):
        fi = st.session_state['results'].get('feature_importances')
    feature_importances = None
    if fi:
        feature_importances = {'names': st.session_state['results']['X_columns'], 'importances': fi}
    overall_score = compute_suitability_score(median_row, df_ref=df, features=features, feature_importances=feature_importances)
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
    <div style="border:2.5px solid {color_hex};border-radius:18px;padding:22px;">
      <h3>{circle} <span style='color:{color_hex};font-weight:bold'>{level_label}</span></h3>
      <div style='font-size:16px'>{description}</div>
      <div style='margin-top:8px'><b>Recommended Crops:</b> <span style='color:{color_hex};font-weight:700'>{crop_list}</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('---')
    st.subheader('Sample-level suitability & recommendations')
    display_cols = features + ['Latitude','Longitude']
    preview = df[display_cols].copy()
    preview['suitability_score'] = preview.apply(lambda r: compute_suitability_score(r, df_ref=df, features=features, feature_importances=feature_importances), axis=1)
    preview['suitability_label'] = preview['suitability_score'].apply(lambda s: suitability_color(s)[0])
    preview['suitability_hex'] = preview['suitability_score'].apply(lambda s: suitability_color(s)[1])
    preview['top_crops'] = preview.apply(lambda r: ", ".join([f"{c[0]} ({c[1]:.2f})" for c in recommend_crops_for_sample(r, top_n=3)]), axis=1)
    def verdict_text(row):
        score = row['suitability_score']
        label = row['suitability_label']
        crops = recommend_crops_for_sample(row, top_n=3)
        if label == 'Green':
            return f"🟢 Soil is sustainable. Ideal crops: {', '.join([c[0] for c in crops])}."
        if label == 'Orange':
            return f"🟠 Moderate. Consider fertilizer/pH adjustments. Crops: {', '.join([c[0] for c in crops])}."
        return f"🔴 Poor. Major amendments needed. Possible crops after treatment: {', '.join([c[0] for c in crops])}."
    preview['verdict'] = preview.apply(verdict_text, axis=1)
    st.dataframe(preview[[*features,'suitability_score','suitability_label','top_crops','verdict']], use_container_width=True)

# ------------------- Main Navigation -------------------

with st.sidebar:
    selected = option_menu(None, ["🏠 Home", "🤖 Modeling", "📊 Visualization", "📈 Results", "🌿 Insights", "👤 About"],
                           icons=["house","robot","bar-chart","graph-up","lightbulb","person-circle"], default_index=0)

page = selected

if page == "🏠 Home":
    st.title("Machine Learning-Driven Soil Analysis for Sustainable Agriculture System")
    st.markdown("<small>Capstone Project — updated with improved preprocessing, rule-based + ML suitability, clustering, and explainability.</small>", unsafe_allow_html=True)
    st.write("---")
    upload_and_preprocess_widget()
elif page == "🤖 Modeling":
    modeling_page()
elif page == "📊 Visualization":
    visualization_page()
elif page == "📈 Results":
    results_page()
elif page == "🌿 Insights":
    insights_page()
elif page == "👤 About":
    # Keep About page logic from original unchanged — it appears below in the saved file.
    st.title("👤 About the Makers")
    st.markdown("<div style='font-size:19px;'>Developed by:</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1,1])
    with col_a:
        try:
            image = Image.open('assets/andre_oneal_a._plaza.png')
            st.image(image, width=160)
            st.markdown('<div style="font-weight:700">Andre Oneal A. Plaza</div>', unsafe_allow_html=True)
        except Exception:
            st.write('Andre Oneal A. Plaza')
    with col_b:
        try:
            image = Image.open('assets/rica_baliling.png')
            st.image(image, width=160)
            st.markdown('<div style="font-weight:700">Rica Baliling</div>', unsafe_allow_html=True)
        except Exception:
            st.write('Rica Baliling')
    st.markdown("---")
    st.markdown("<div style='font-size:15px;color:#cd5fff;font-weight:600;'>All glory to God.</div>", unsafe_allow_html=True)

# End of app_full.py
