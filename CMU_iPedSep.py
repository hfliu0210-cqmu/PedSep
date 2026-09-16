import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import shap
import xgboost as xgb
import joblib
import networkx as nx
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import streamlit as st
from PIL import Image

st.set_page_config(page_title=" Prediction Model For Pediatric Sepsis CMU‑iPedSep", layout="wide")
plt.switch_backend("Agg")

CONFIG = {
    "dpi": 300,
    "font_en": "Times New Roman",
    "font_zh": "Arial Unicode MS",
    "random_state": 42,
    "model_path": "ipedsep_model.joblib",
    "explainer_path": "ipedsep_explainer.joblib",
    "feature_map": [
        ("Age", "Age (month)", False),
        ("Hemoglobin", "Hemoglobin (g/L)", False),
        ("Lymphocyte", "Lymphocyte (10^9/L)", False),
        ("Monocyte", "Monocyte (10^9/L)", False),
        ("MCHC", "MCHC (g/L)", False),
        ("RDW‑CV", "RDW‑CV (%)", False),
        ("Neutrophil‑Lymphocyte Ratio (NLR)", False),
        ("Underlying condition", "Underlying condition (prematurity, chronic diseases, severe malnutrition; 1=Yes, 0=No)", True)
    ],
    "fig7_max_features": 8,
    "fig9_figsize": (20, 4.5),
}

plt.rcParams['font.sans-serif'] = [CONFIG["font_en"], CONFIG["font_zh"], "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="ticks")

def fig2pil(fig):
    buf = BytesIO()
    fig.savefig(buf, dpi=CONFIG["dpi"], format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


class CMU_iPedSep_Model:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = [item[0] for item in CONFIG["feature_map"]]
        self.train_median = {}
        self.var_filter = None
        self.scaler = None
        self.shap_interaction = None
        self.X_train_processed = None

    def load_model(self):
        self.model = joblib.load(CONFIG["model_path"])
        save_data = joblib.load(CONFIG["explainer_path"])
        self.explainer = save_data["explainer"]
        self.train_median = save_data["train_median"]
        self.var_filter = save_data["var_filter"]
        self.scaler = save_data["scaler"]
        self.shap_interaction = save_data["shap_interaction"]
        self.X_train_processed = save_data["X_train_processed"]

    def plot_fig7_network(self):
        mean_abs_shap = np.abs(self.explainer.shap_values(self.X_train_processed, check_additivity=False)).mean(axis=0)
        sort_idx = np.argsort(mean_abs_shap)[::-1]
        top_n = min(CONFIG["fig7_max_features"], len(self.feature_names))
        top_idx = sort_idx[:top_n]
        top_feat_names = [self.feature_names[i] for i in top_idx]

        G = nx.Graph()
        for idx, name in zip(top_idx, top_feat_names):
            G.add_node(name, weight=mean_abs_shap[idx])
        for i in range(top_n):
            for j in range(i+1, top_n):
                ii = top_idx[i]
                jj = top_idx[j]
                inter = np.abs(self.shap_interaction[:, ii, jj]).mean()
                G.add_edge(top_feat_names[i], top_feat_names[j], weight=inter)

        pos = nx.circular_layout(G)
        node_w = [G.nodes[n]["weight"] for n in G.nodes]
        edge_w = [G[u][v]["weight"] for u, v in G.edges]
        node_sizes = 500 + 1800 * (np.array(node_w) / (np.max(node_w)+1e-8))
        edge_widths = 1 + 6 * (np.array(edge_w)/(np.max(edge_w)+1e-8))

        fig, ax = plt.subplots(figsize=(9,9))
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_w,
                               edge_cmap=mcolors.LinearSegmentedColormap.from_list("",["#f8ccd2","#c51e36"]),
                               alpha=0.7, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_w,
                               cmap=mcolors.LinearSegmentedColormap.from_list("",["#e2edf5","#273b52"]),
                               edgecolors="gray", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=11, fontfamily=CONFIG["font_en"], ax=ax)
        ax.set_title("Fig7 Feature Impact & Interaction Network", fontsize=15, fontfamily=CONFIG["font_en"])
        ax.axis("off")
        return fig2pil(fig)

    def plot_fig9_force(self, input_dict):
        input_df = pd.DataFrame([input_dict])
        # 不再自动填充中位数，缺失校验放在main函数层
        X_var = self.var_filter.transform(input_df)
        X_scaled = self.scaler.transform(X_var)
        X_proc = pd.DataFrame(X_scaled, columns=self.feature_names)
        shap_vals = self.explainer.shap_values(X_proc, check_additivity=False)
        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[-1]

        feat_series = pd.Series(X_proc.iloc[0,:].round(3).values, index=self.feature_names)
        fig = shap.force_plot(base_val, shap_vals[0], feat_series,
                              matplotlib=True, show=False, figsize=CONFIG["fig9_figsize"])
        ax = fig.gca()
        for t in ax.texts:
            t.set_fontfamily(CONFIG["font_en"])
        return fig2pil(fig)


def main():
    # Top banner
    top_col1, top_col2 = st.columns([0.08, 0.92])
    with top_col1:
        if os.path.exists("logo1.png"):
            st.image("logo1.png", use_container_width=True)
    with top_col2:
        st.markdown("# Prediction Model For Pediatric Sepsis")
        st.markdown("## CMU‑iPedSep")
    st.divider()

    model = CMU_iPedSep_Model()
    try:
        model.load_model()
        st.success("✅ Prediction model loaded. Please input clinical indicators for analysis.")
    except FileNotFoundError:
        st.error("❌ Model file missing: ipedsep_model.joblib / ipedsep_explainer.joblib")
        st.stop()

    st.divider()
    # Main two‑column layout
    left_col, right_col = st.columns([0.48, 0.52])

    with left_col:
        st.header("Input Panel")
        st.markdown("### Enter patient clinical indicators (All 8 fields cannot be empty)")
        input_data = {}
        in_col1, in_col2 = st.columns(2)
        half = len(CONFIG["feature_map"]) // 2
        with in_col1:
            for raw_name, show_name, is_bin in CONFIG["feature_map"][:half]:
                if is_bin:
                    input_data[raw_name] = st.number_input(label=show_name, value=0, min_value=0, max_value=1, step=1)
                else:
                    input_data[raw_name] = st.number_input(label=show_name, value=None, step=0.01)
        with in_col2:
            for raw_name, show_name, is_bin in CONFIG["feature_map"][half:]:
                if is_bin:
                    input_data[raw_name] = st.number_input(label=show_name, value=0, min_value=0, max_value=1, step=1)
                else:
                    input_data[raw_name] = st.number_input(label=show_name, value=None, step=0.01)

        btn_fig7 = st.button("Generate Fig7 Feature Interaction Network")
        btn_fig9 = st.button("Generate Fig9 SHAP Force Plot")

    with right_col:
        st.header("Output Panel")

        # --------缺失值校验函数--------
        def check_all_input_valid():
            missing_list = []
            for raw_name, show_name, _ in CONFIG["feature_map"]:
                val = input_data[raw_name]
                # 判断空 / nan
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    missing_list.append(show_name)
            if len(missing_list) > 0:
                st.error(f"❌ Missing value detected: {', '.join(missing_list)}. All 8 indicators must be filled, cannot be empty.")
                return False
            return True

        if btn_fig7:
            if check_all_input_valid():
                with st.spinner("Rendering Fig7 ..."):
                    fig7_img = model.plot_fig7_network()
                st.subheader("Fig7 Feature Interaction Network")
                st.image(fig7_img, use_container_width=True)
                buf7 = BytesIO()
                fig7_img.save(buf7, format="png")
                st.download_button(label="Download Fig7 PNG",
                                   data=buf7.getvalue(),
                                   file_name="Fig7_CMU‑iPedSep.png",
                                   mime="image/png")

        if btn_fig9:
            if check_all_input_valid():
                with st.spinner("Calculating SHAP and rendering Fig9 ..."):
                    fig9_img = model.plot_fig9_force(input_data)
                st.subheader("Fig9 SHAP Force Plot")
                st.image(fig9_img, use_container_width=True)
                buf9 = BytesIO()
                fig9_img.save(buf9, format="png")
                st.download_button(label="Download Fig9 PNG",
                                   data=buf9.getvalue(),
                                   file_name="Fig9_CMU‑iPedSep.png",
                                   mime="image/png")


if __name__ == "__main__":
    main()
