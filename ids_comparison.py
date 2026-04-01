"""
IoT Intrusion Detection: Rule-Based vs Machine Learning
========================================================
Dataset: RT-IoT2022  (UCI ML Repository, id=942)
         Real-Time IoT traffic captured with Zeek + Flowmeter
         ~112,000 rows · 83 features · 12 attack/traffic classes
         CC BY 4.0 — cite: S., B. & Nagapadma, R. (2023)

Quick-start
-----------
    pip3 install ucimlrepo scikit-learn matplotlib seaborn pandas numpy
    python ids_comparison.py

The script auto-downloads RT-IoT2022 on first run via ucimlrepo.
No manual CSV wrangling required.
"""

import numpy as np
import pandas as pd
import warnings
import json
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Scikit-learn ─────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# ── Plotting ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


# ═════════════════════════════════════════════════════════════════════════════
# 1.  RT-IoT2022 DATASET LOADER
# ═════════════════════════════════════════════════════════════════════════════

# Map the 12 fine-grained RT-IoT2022 traffic types to 4 parent categories
# that mirror classic IDS taxonomy and keep confusion matrices readable.
LABEL_MAP = {
    # Normal IoT device traffic
    "MQTT_Publish":               "Normal",
    "Thing_Speak":                "Normal",
    "Wipro_bulb":                 "Normal",
    "Amazon-Alexa":               "Normal",
    # DoS / DDoS
    "DOS_SYN_Hping":              "DoS",
    "DDOS_Slowloris":             "DoS",
    # Reconnaissance / Port Scan
    "NMAP_UDP_SCAN":              "Recon",
    "NMAP_XMAS_TREE_SCAN":        "Recon",
    "NMAP_OS_DETECTION":          "Recon",
    "NMAP_TCP_scan":              "Recon",
    "NMAP_FIN_SCAN":              "Recon",
    # Brute Force
    "Metasploit_Brute_Force_SSH": "BruteForce",
}

# Ordered list — used for confusion-matrix axes and per-class F1
CLASS_NAMES = ["Normal", "DoS", "Recon", "BruteForce"]


def load_rt_iot2022() -> pd.DataFrame:
    """
    Load RT-IoT2022 from the UCI ML Repository via ucimlrepo.

    Returns a DataFrame with numeric feature columns plus:
      'attack_type'  : human-readable category string
      'label'        : integer class index (0-3)
    """
    print("    Fetching RT-IoT2022 from UCI ML Repository …")
    print("    (first run downloads ~15 MB; cached afterwards)")

    try:
        from ucimlrepo import fetch_ucirepo
        dataset    = fetch_ucirepo(id=942)
        X          = dataset.data.features.copy()
        y_raw      = dataset.data.targets.copy()
        target_col = y_raw.columns[0]          # 'Attack_type'
        df         = X.copy()
        df["attack_type_raw"] = y_raw[target_col].values

    except Exception as exc:
        raise RuntimeError(
            f"Could not load RT-IoT2022: {exc}\n"
            "Run:  pip install ucimlrepo"
        ) from exc

    # Map fine-grained labels to 4 categories
    df["attack_type"] = df["attack_type_raw"].map(LABEL_MAP)
    n_unknown = df["attack_type"].isna().sum()
    if n_unknown:
        unseen = df.loc[df["attack_type"].isna(), "attack_type_raw"].unique()
        print(f"    [!] Dropping {n_unknown} rows with unmapped labels: {unseen}")
    df = df.dropna(subset=["attack_type"]).reset_index(drop=True)

    # Encode to integers
    le = LabelEncoder()
    le.classes_ = np.array(CLASS_NAMES)
    df["label"] = le.transform(df["attack_type"])

    # Remove the raw label column
    df = df.drop(columns=["attack_type_raw"])

    # Clean: inf / NaN → 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Drop any non-numeric columns that aren't our two label cols
    non_num = df.select_dtypes(exclude=[np.number]).columns.tolist()
    to_drop = [c for c in non_num if c not in ("attack_type", "label")]
    if to_drop:
        print(f"    [!] Dropping non-numeric columns: {to_drop}")
        df = df.drop(columns=to_drop)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2.  RULE-BASED IDS  (traditional / baseline)
#
#  RT-IoT2022 Zeek feature names used by the rules:
#    orig_pkts   — packets sent by the originator (attacker side)
#    resp_pkts   — packets sent by the responder
#    orig_bytes  — bytes from originator
#    resp_bytes  — bytes from responder
#    duration    — flow duration in seconds
#    id.resp_p   — destination port
# ═════════════════════════════════════════════════════════════════════════════

class RuleBasedIDS:
    """
    Signature / threshold-based IDS calibrated to RT-IoT2022.

    Rule priority (first match wins):
      1. DoS        — high packet count, short duration, tiny responses
      2. BruteForce — repeated small flows to SSH port 22
      3. Recon      — tiny payloads, very short probe flows
      4. Normal     — everything else
    """

    T = {
        "dos_orig_pkts_min":    500,   # SYN flood: thousands of tiny pkts
        "dos_duration_max":     2.0,   # seconds — floods are short bursts
        "dos_resp_bytes_max":   500,   # server sends almost nothing back

        "bf_port":              22,    # SSH port
        "bf_orig_bytes_max":    2000,  # small payload per login attempt
        "bf_duration_max":      5.0,   # seconds

        "recon_orig_bytes_max": 200,   # Nmap probe packets are tiny
        "recon_duration_max":   0.5,   # sub-second flows
        "recon_resp_pkts_max":  3,     # minimal response to probes
    }

    # Primary column names as they appear in RT-IoT2022 after Zeek extraction
    _COLS = {
        "orig_pkts":  "orig_pkts",
        "resp_pkts":  "resp_pkts",
        "orig_bytes": "orig_bytes",
        "resp_bytes": "resp_bytes",
        "duration":   "duration",
        "dest_port":  "id.resp_p",
    }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Resolve actual column names once (graceful if a column is missing)
        self._resolved = {
            k: (v if v in X.columns else None)
            for k, v in self._COLS.items()
        }
        return np.array([self._classify(row) for _, row in X.iterrows()])

    def _get(self, row, key, default=0.0):
        col = self._resolved.get(key)
        if col is None:
            return default
        v = row.get(col, default)
        return default if pd.isna(v) else float(v)

    def _classify(self, row) -> int:
        T = self.T
        orig_pkts  = self._get(row, "orig_pkts")
        resp_bytes = self._get(row, "resp_bytes")
        orig_bytes = self._get(row, "orig_bytes")
        resp_pkts  = self._get(row, "resp_pkts")
        duration   = self._get(row, "duration")
        dest_port  = self._get(row, "dest_port", -1)

        # DoS rule
        if (orig_pkts  > T["dos_orig_pkts_min"] and
                duration   < T["dos_duration_max"] and
                resp_bytes < T["dos_resp_bytes_max"]):
            return 1  # DoS

        # Brute-force rule
        if (dest_port  == T["bf_port"] and
                orig_bytes < T["bf_orig_bytes_max"] and
                duration   < T["bf_duration_max"]):
            return 3  # BruteForce

        # Recon rule
        if (orig_bytes < T["recon_orig_bytes_max"] and
                duration   < T["recon_duration_max"] and
                resp_pkts  < T["recon_resp_pkts_max"]):
            return 2  # Recon

        return 0  # Normal


# ═════════════════════════════════════════════════════════════════════════════
# 3.  EVALUATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    labels = list(range(len(CLASS_NAMES)))

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted",
                           labels=labels, zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted",
                        labels=labels, zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted",
                    labels=labels, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=labels)
    f1pc = f1_score(y_true, y_pred, average=None,
                    labels=labels, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"\n  Per-class F1:")
    for cls, score in zip(CLASS_NAMES, f1pc):
        bar = "█" * int(score * 20)
        print(f"    {cls:<14} {score:.3f}  {bar}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES,
                                labels=labels, zero_division=0))

    return {
        "model": model_name,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": cm, "f1_per_class": f1pc,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4.  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "Rule-Based IDS":    "#e74c3c",
    "Decision Tree":     "#f39c12",
    "Random Forest":     "#27ae60",
    "Gradient Boosting": "#2980b9",
}
BG    = "#0d1117"
PANEL = "#161b22"
TEXT  = "#c9d1d9"
GRID  = "#21262d"


def _style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)


def plot_results(results: list, output_path: str):
    models  = [r["model"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1"]
    mlabels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors  = [PALETTE.get(m, "#8b949e") for m in models]

    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    fig.suptitle("RT-IoT2022  |  Rule-Based IDS vs Machine Learning",
                 fontsize=17, color=TEXT, fontweight="bold", y=0.98)

    gs = GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.35,
                  top=0.93, bottom=0.05, left=0.06, right=0.97)

    # Row 0 — metric bar charts
    for col, (metric, lbl) in enumerate(zip(metrics, mlabels)):
        ax   = fig.add_subplot(gs[0, col])
        vals = [r[metric] for r in results]
        bars = ax.bar(models, vals, color=colors, width=0.55, zorder=3)
        ax.set_ylim(0, 1.12)
        ax.set_title(lbl, fontsize=11, pad=6)
        ax.set_ylabel("Score", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        _style(ax)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, color=TEXT)

    # Row 1 — normalised confusion matrices
    for col, result in enumerate(results):
        ax  = fig.add_subplot(gs[1, col])
        cm  = result["confusion_matrix"]
        cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        sns.heatmap(cmn, annot=True, fmt=".2f",
                    cmap=sns.color_palette("mako", as_cmap=True),
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, cbar=False, linewidths=0.3, linecolor=BG,
                    annot_kws={"size": 8}, vmin=0, vmax=1)
        ax.set_title(result["model"], fontsize=9, pad=4)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("Actual",    fontsize=7)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)
        _style(ax)

    # Row 2 left — per-class F1 grouped bars
    ax_f1 = fig.add_subplot(gs[2, :2])
    x     = np.arange(len(CLASS_NAMES))
    width = 0.18
    for i, result in enumerate(results):
        offset = (i - len(results)/2 + 0.5) * width
        ax_f1.bar(x + offset, result["f1_per_class"], width,
                  label=result["model"], color=colors[i], alpha=0.88, zorder=3)
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax_f1.set_ylabel("F1-Score", fontsize=9)
    ax_f1.set_title("Per-Class F1 Score Comparison", fontsize=11)
    ax_f1.set_ylim(0, 1.12)
    ax_f1.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID,
                 labelcolor=TEXT, loc="lower right")
    ax_f1.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    _style(ax_f1)

    # Row 2 right — radar / spider chart
    ax_r = fig.add_subplot(gs[2, 2:], polar=True)
    ax_r.set_facecolor(PANEL)
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(mlabels, color=TEXT, fontsize=8)
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                          color="#8b949e", fontsize=6)
    ax_r.grid(color=GRID, linewidth=0.4)
    ax_r.spines["polar"].set_edgecolor(GRID)
    ax_r.set_title("Radar: All Metrics", fontsize=11, color=TEXT, pad=12)
    for i, result in enumerate(results):
        v = [result[m] for m in metrics] + [result[metrics[0]]]
        ax_r.plot(angles, v, linewidth=1.8, color=colors[i],
                  label=result["model"])
        ax_r.fill(angles, v, alpha=0.07, color=colors[i])
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18),
                fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  [✓] Comparison chart → {output_path}")


def plot_feature_importance(rf_model, feature_names: list, output_path: str):
    imp     = rf_model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:15]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.barh(range(15), imp[top_idx][::-1],
            color=plt.cm.viridis(np.linspace(0.2, 0.85, 15)),
            edgecolor=BG, linewidth=0.5)
    ax.set_yticks(range(15))
    ax.set_yticklabels([feature_names[i] for i in top_idx[::-1]],
                       fontsize=8, color=TEXT)
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=9, color=TEXT)
    ax.set_title("Random Forest — Top 15 Feature Importances  (RT-IoT2022)",
                 fontsize=11, color=TEXT, pad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.tick_params(colors=TEXT)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  [✓] Feature importance → {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
from pathlib import Path 
def main():
    OUT = Path('outputs')
    OUT.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  IoT IDS COMPARISON  ·  RT-IoT2022 dataset")
    print("="*60)

    # 5.1  Load
    print("\n[1] Loading RT-IoT2022 …")
    df = load_rt_iot2022()

    print(f"\n    Total samples : {len(df):,}")
    print(f"    Feature cols  : {df.shape[1] - 2}")
    print(f"\n    Class distribution:")
    for cls, cnt in df["attack_type"].value_counts().items():
        print(f"      {cls:<14} {cnt:6,}  ({cnt/len(df)*100:.1f}%)")

    # 5.2  Split
    feature_cols = [c for c in df.columns if c not in ("attack_type", "label")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # DataFrame version of test set for the rule-based IDS
    X_test_df  = pd.DataFrame(X_test, columns=feature_cols)

    print(f"\n    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 5.3  Rule-Based IDS
    print("\n[2] Running Rule-Based IDS …")
    rule_ids    = RuleBasedIDS()
    y_pred_rule = rule_ids.predict(X_test_df)
    r_rule      = compute_metrics(y_test, y_pred_rule, "Rule-Based IDS")

    # 5.4  ML models
    print("\n[3] Training ML models …")
    ml_models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=20,
            class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42),
    }

    ml_results = []
    rf_model   = None

    for name, model in ml_models.items():
        print(f"\n    Training {name} …")
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        result = compute_metrics(y_test, y_pred, name)
        ml_results.append(result)

        cv = cross_val_score(model, X_train_sc, y_train,
                             cv=5, scoring="f1_weighted", n_jobs=-1)
        print(f"    5-fold CV F1: {cv.mean():.4f} ± {cv.std():.4f}")

        if name == "Random Forest":
            rf_model = model

    # 5.5  Summary table
    all_results = [r_rule] + ml_results

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10}"
          f" {'Recall':>8} {'F1':>8}")
    print("  " + "-"*58)
    for r in all_results:
        print(f"  {r['model']:<22} {r['accuracy']:>9.4f}"
              f" {r['precision']:>10.4f} {r['recall']:>8.4f}"
              f" {r['f1']:>8.4f}")

    best = max(all_results, key=lambda x: x["f1"])
    print(f"\n  ★  Best: {best['model']}  (F1 = {best['f1']:.4f})")

    # 5.6  Export JSON
    export = []
    for r in all_results:
        export.append({
            "model":        r["model"],
            "accuracy":     round(r["accuracy"],  4),
            "precision":    round(r["precision"], 4),
            "recall":       round(r["recall"],    4),
            "f1":           round(r["f1"],        4),
            "f1_per_class": {
                cls: round(float(s), 4)
                for cls, s in zip(CLASS_NAMES, r["f1_per_class"])
            },
        })
    with open(OUT / "metrics_summary.json", "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n  [✓] Metrics JSON → {OUT}/metrics_summary.json")

    # 5.7  Plots
    print("\n[4] Generating visualisations …")
    plot_results(all_results, str(OUT / "ids_comparison.png"))
    plot_feature_importance(rf_model, feature_cols,
                            str(OUT / "feature_importance.png"))

    print("\n[✓] Pipeline complete.\n")
    return all_results


if __name__ == "__main__":
    main()
