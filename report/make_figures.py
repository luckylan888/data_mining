"""生成报告用窄栏图（ACL 双栏版式，单栏宽 8 cm），并跑两个小实验：
1) XOR：单层线性模型 vs 含隐层 MLP；
2) 重复划分下方差对比：单棵决策树 vs 随机森林。
输出：report/figs/*.png、report/extra_numbers.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "legend.frameon": False,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})
COL = 3.15  # 英寸，约 8 cm

PRETTY = {
    "基线(全预测不流失)": "Baseline",
    "朴素贝叶斯": "Naive Bayes",
    "SVM-线性核": "SVM (linear)",
    "SVM-RBF核": "SVM (RBF)",
    "MLP": "MLP",
    "决策树": "Decision Tree",
    "随机森林": "Random Forest",
    "GBDT(HistGB)": "GBDT (HistGB)",
    "GBDT+加权+等渗校准": "GBDT + isotonic",
    "CatBoost(原生类别)": "CatBoost",
}
COLORS = {
    "GBDT+加权+等渗校准": "#4E79A7",
    "CatBoost(原生类别)": "#59A14F",
    "随机森林": "#2E5EAA",
    "SVM-线性核": "#D1495B",
}


def save(fig, name):
    fig.savefig(OUT / name, dpi=400)
    plt.close(fig)


scores = pd.read_csv(RES / "model_scores.csv")
biz = pd.read_csv(RES / "business_metrics.csv")
prob = pd.read_csv(RES / "test_probabilities.csv")
y = prob["y"].to_numpy()

# 图1：类别不平衡与合同类型
raw = pd.read_csv(ROOT.parent / "data" / "telco_churn.csv").dropna(subset=["Churn"])
raw["churn"] = (raw["Churn"] == "Yes").astype(int)

fig, axes = plt.subplots(1, 2, figsize=(COL, 2.25))
rate = raw.groupby("Contract")["churn"].mean().reindex(["Month-to-month", "One year", "Two year"])
axes[0].bar(range(3), rate.values * 100, color=["#E15759", "#F28E2B", "#4E79A7"],
            width=0.62, edgecolor="black", linewidth=0.5)
axes[0].set_xticks(range(3), ["Monthly", "1 year", "2 years"])
axes[0].set_ylabel("Churn rate (%)")
axes[0].set_title("(a) Churn rate by contract")
for i, v in enumerate(rate.values * 100):
    axes[0].text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=6.5)
axes[0].set_ylim(0, 50)

for label, color in [(0, "#4E79A7"), (1, "#E15759")]:
    axes[1].hist(raw.loc[raw["churn"] == label, "tenure"], bins=18, density=True,
                 alpha=0.55, color=color, label=f"Churn={label}", edgecolor="white", linewidth=0.3)
axes[1].set_xlabel("Tenure (months)")
axes[1].set_ylabel("Density")
axes[1].set_title("(b) Tenure by class")
axes[1].legend(loc="upper right")
fig.tight_layout(pad=0.3)
save(fig, "fig1_eda.png")

# 图2：模型对比（PR-AUC 与 AUC，误差棒为 95% bootstrap CI）
sub = scores[scores["model"] != "基线(全预测不流失)"].sort_values("pr_auc")
fig, ax = plt.subplots(figsize=(COL, 2.9))
pos = np.arange(len(sub))
ax.errorbar(sub["pr_auc"], pos + 0.13, xerr=[sub["pr_auc"] - sub["pr_lo"], sub["pr_hi"] - sub["pr_auc"]],
            fmt="o", ms=3.4, color="#D1495B", ecolor="#D1495B", elinewidth=0.8, capsize=1.6, label="PR-AUC")
ax.errorbar(sub["auc"], pos - 0.13, xerr=[sub["auc"] - sub["auc_lo"], sub["auc_hi"] - sub["auc"]],
            fmt="s", ms=3.2, color="#4E79A7", ecolor="#4E79A7", elinewidth=0.8, capsize=1.6, label="ROC-AUC")
ax.set_yticks(pos, [PRETTY[m] for m in sub["model"]])
ax.set_xlim(0.45, 0.90)
ax.set_xlabel("Score with 95% bootstrap CI")
ax.legend(loc="lower right", ncol=2)
ax.grid(axis="x", color="#E5E5E5", lw=0.5)
fig.tight_layout(pad=0.3)
save(fig, "fig2_models.png")

# 图3：ROC 与 PR（四个代表模型）
sel = ["SVM-线性核", "随机森林", "GBDT+加权+等渗校准", "CatBoost(原生类别)"]
fig, axes = plt.subplots(2, 1, figsize=(COL, 4.0))
for name in sel:
    p = prob[name].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, lw=1.0, color=COLORS[name], label=f"{PRETTY[name]} ({roc_auc_score(y, p):.3f})")
    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].plot(rec, prec, lw=1.0, color=COLORS[name],
                 label=f"{PRETTY[name]} ({average_precision_score(y, p):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=0.6)
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("(a) ROC")
axes[0].legend(loc="lower right")
axes[0].grid(color="#EEEEEE", lw=0.45)

axes[1].axhline(y.mean(), color="gray", ls="--", lw=0.6)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("(b) Precision-recall")
axes[1].legend(loc="upper right")
axes[1].grid(color="#EEEEEE", lw=0.45)
fig.tight_layout(pad=0.3)
save(fig, "fig3_roc_pr.png")

# 图4：SHAP 全局重要性
shap_imp = pd.read_csv(RES / "shap_importance.csv", index_col=0)["mean_abs_shap"]
top = shap_imp.head(8)[::-1]
fig, ax = plt.subplots(figsize=(COL, 2.4))
ax.barh(range(len(top)), top.values, color="#59A14F", edgecolor="black", linewidth=0.4, height=0.62)
ax.set_yticks(range(len(top)), top.index)
ax.set_xlabel("mean |SHAP| (log-odds)")
ax.set_title("Global attribution (CatBoost)")
ax.grid(axis="x", color="#E5E5E5", lw=0.5)
fig.tight_layout(pad=0.3)
save(fig, "fig4_shap.png")

# 图5：利润-容量曲线（需要测试集月费，重建同一划分）
clean = pd.read_csv(ROOT.parent / "data" / "telco_churn.csv")
clean = clean.dropna(subset=["Churn"])
clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"].astype(str).str.strip(), errors="coerce")
clean.loc[(clean["tenure"] == 0) & clean["TotalCharges"].isna(), "TotalCharges"] = 0
y_all = (clean["Churn"] == "Yes").astype(int)
X_all = clean.drop(columns=["Churn", "customerID"])
Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.25, stratify=y_all, random_state=42)
assert np.allclose(yte.to_numpy(), y), "测试集标签与流水线不一致"
monthly = Xte["MonthlyCharges"].to_numpy()
arr = 1 - ytr.mean()

caps = np.linspace(0.05, 0.60, 23)
fig, ax = plt.subplots(figsize=(COL, 2.35))
for name in ["随机森林", "GBDT+加权+等渗校准", "CatBoost(原生类别)"]:
    p = prob[name].to_numpy()
    profits = []
    for c in caps:
        pred = np.zeros(len(p), dtype=int)
        pred[np.argsort(-p)[: int(round(c * len(p)))]] = 1
        profits.append((((yte == 1) & (pred == 1)) * arr * 24 * monthly - pred * 50).sum() / len(p) * 1000)
    ax.plot(caps * 100, profits, "o-", ms=2.2, lw=1.0, color=COLORS[name], label=PRETTY[name])
ax.axvline(20, color="gray", ls="--", lw=0.7)
ax.set_xlabel("Contacted fraction (%)")
ax.set_ylabel("Profit (USD / 1,000)")
ax.set_title("Profit vs. capacity")
ax.legend(loc="lower right")
ax.grid(color="#EEEEEE", lw=0.45)
fig.tight_layout(pad=0.3)
save(fig, "fig5_profit.png")

# 小实验 1：XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])
lr_acc = LogisticRegression(C=10, max_iter=1000).fit(X_xor, y_xor).score(X_xor, y_xor)
mlp_acc = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh", max_iter=5000,
                        random_state=0).fit(X_xor, y_xor).score(X_xor, y_xor)

# 小实验 2：固定测试集，重复抽取训练子集，观察模型本身的波动（避免测试集抽样噪声掩盖方差差异）
from sklearn.model_selection import StratifiedShuffleSplit

X_enc = pd.get_dummies(X_all, drop_first=True, dtype=float)
Xtr_enc, Xte_enc, ytr_enc, yte_enc = train_test_split(
    X_enc, y_all, test_size=0.25, stratify=y_all, random_state=42)
auc_dt, auc_rf = [], []
sss = StratifiedShuffleSplit(n_splits=10, train_size=0.75, random_state=7)
for tr_idx, _ in sss.split(Xtr_enc, ytr_enc):
    Xa, ya = Xtr_enc.iloc[tr_idx], ytr_enc.iloc[tr_idx]
    dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xa, ya)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42).fit(Xa, ya)
    auc_dt.append(roc_auc_score(yte_enc, dt.predict_proba(Xte_enc)[:, 1]))
    auc_rf.append(roc_auc_score(yte_enc, rf.predict_proba(Xte_enc)[:, 1]))

extra = {
    "xor_linear_acc": float(lr_acc),
    "xor_mlp_acc": float(mlp_acc),
    "dt_auc_mean": float(np.mean(auc_dt)),
    "dt_auc_std": float(np.std(auc_dt, ddof=1)),
    "rf_auc_mean": float(np.mean(auc_rf)),
    "rf_auc_std": float(np.std(auc_rf, ddof=1)),
    "arr": float(arr),
}
with open(Path(__file__).resolve().parent / "extra_numbers.json", "w", encoding="utf-8") as f:
    json.dump(extra, f, ensure_ascii=False, indent=2)
print(json.dumps(extra, indent=2))
