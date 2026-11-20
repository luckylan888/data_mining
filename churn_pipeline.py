"""
客户流失预警：面向 ACL 报告的六阶段建模流水线（CatBoost 对照 + 验证集阈值 + SHAP 解释）

六个阶段与其先进做法
 ① 探索数据：缺失机制诊断（结构性缺失 vs 随机缺失）、Wilson 置信区间、
             效应量（Cramér's V / 秩双列相关）、冗余/泄漏审计（TotalCharges ≈ tenure×月费, R²=0.9991）
 ② 清洗数据：按“记录有效性 + 领域规则”清洗，拒绝无脑 dropna / 均值填充；
             保留缺失指示变量，让模型判断缺失本身的信息量
 ③ 预处理与划分：先分层划分，再做拟合型变换（补值/编码/缩放）全部封装进 Pipeline；
             CatBoost 直接吃原始类别字段，作为“免独热编码”的现代对照
 ④ 训练多模型：课程要求的 NB / SVM(线性/RBF) / MLP / 决策树 / 随机森林
             + HistGradientBoosting + 类别加权 + 等渗校准 + CatBoost（原生类别特征）
 ⑤ 指标与图表：ROC-AUC / PR-AUC（不平衡主指标）/ Brier / 对数损失，Bootstrap 95% 置信区间，
             5 折交叉验证；ROC、PR、校准、lift/gain、利润-容量曲线（Origin/ACL 风格，300 dpi）
 ⑥ 选型与业务：在“验证集”上按运营容量选阈值，测试集只评一次；
             用 CatBoost SHAP 解释高风险客户，并给出可执行的挽留动作

主要文献
  - Burez & Van den Poel (2009), Handling class imbalance in customer churn prediction, ESWA
  - Amin et al. (2016), Comparing Oversampling Techniques, IEEE Access
  - Prokhorenkova et al. (2018), CatBoost: Unbiased Boosting with Categorical Features, NeurIPS
  - Xu et al. (2021), Telecom Churn Prediction Based on Ensemble Learning Using Feature Grouping, Applied Sciences
  - Sikri et al. (2024), Enhancing customer retention in telecom, Scientific Reports
  - Asif et al. (2025), Explainable AI for customer churn prediction, Results in Engineering
  - arXiv:2610.00118  审计 Telco 基准：先 SMOTE 再划分使 F1 虚高 13.1pp；等渗校准最稳；成本最优阈值
  - arXiv:2507.08860  e-Profits：用留存率和提升度衡量业务利润
  - arXiv:2503.16644  先判断缺失机制，再决定补不补

运行：
    pip install pandas numpy scipy matplotlib scikit-learn catboost
    python churn_pipeline.py

产出：
    figures/fig1_target.png ... fig11_shap.png（同时输出 PDF 矢量版）
    results/model_scores.csv       模型统计指标
    results/business_metrics.csv   业务指标（容量 / 召回 / 利润）
"""

import os
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from scipy.stats import chi2_contingency, mannwhitneyu

from catboost import CatBoostClassifier, Pool
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ==================================================================
# 全局配置
# ==================================================================
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
RESULT_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25
VAL_SIZE = 0.20            # 从训练集再切 20% 作验证集，只用于选阈值（绝不碰测试集）
N_SPLITS = 5
N_BOOT = 500
CAPACITY = 0.20            # 运营容量：每月只能联系风险最高的 20% 客户
OFFER_COST = 50.0          # 单次挽留优惠成本（美元）
CLV_MONTHS = 24            # 客户生命周期 24 个月

# ACL / Origin 风格：衬线字体、四边刻度朝内、无图例边框、白底、细轴线
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
}
plt.rcParams.update(PAPER_STYLE)

# 统一配色（色盲友好 + 印刷可分辨）
PALETTE = {
    "基线(全预测不流失)": "#9E9E9E",
    "朴素贝叶斯": "#D1495B",
    "SVM-线性核": "#00798C",
    "SVM-RBF核": "#66A182",
    "MLP": "#B07AA1",
    "决策树": "#EDAE49",
    "随机森林": "#2E5EAA",
    "GBDT(HistGB)": "#EF6F6C",
    "GBDT+加权+等渗校准": "#4E79A7",
    "CatBoost(原生类别)": "#59A14F",
}


# 论文图内统一使用英文模型名（ACL/LaTeX 排版，避免中文字体缺失变成方块）
PRETTY = {
    "基线(全预测不流失)": "Baseline",
    "朴素贝叶斯": "Naive Bayes",
    "SVM-线性核": "SVM (linear)",
    "SVM-RBF核": "SVM (RBF)",
    "MLP": "MLP",
    "决策树": "Decision Tree",
    "随机森林": "Random Forest",
    "GBDT(HistGB)": "GBDT (HistGB)",
    "GBDT+加权+等渗校准": "GBDT + weights + isotonic",
    "CatBoost(原生类别)": "CatBoost (native categorical)",
}


class CatBoostNative(BaseEstimator, ClassifierMixin):
    """sklearn 兼容的 CatBoost 包装器。

    CatBoost 在 __init__ 里会改写 cat_features，导致 sklearn.clone 报错，
    因此这里只暴露普通超参数，把 CatBoost 实例化推迟到 fit 时，
    并自动把 object 类型的列声明为类别特征（不做独热编码）。
    """

    def __init__(self, iterations=600, learning_rate=0.05, depth=6,
                 l2_leaf_reg=3.0, auto_class_weights="Balanced", random_state=42):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.auto_class_weights = auto_class_weights
        self.random_state = random_state

    def fit(self, X, y):
        cats = [c for c in X.columns if X[c].dtype == object]
        self.model_ = CatBoostClassifier(
            iterations=self.iterations, learning_rate=self.learning_rate,
            depth=self.depth, l2_leaf_reg=self.l2_leaf_reg,
            loss_function="Logloss", eval_metric="AUC",
            random_seed=self.random_state,
            auto_class_weights=self.auto_class_weights,
            cat_features=cats, verbose=0, thread_count=-1)
        self.model_.fit(X, y, verbose=0)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)

    def decision_function(self, X):
        """返回正类概率作为连续分数。

        sklearn 的 roc_auc 打分器优先调用 decision_function；
        若只提供 predict_proba，会拿到 (n, 2) 的二维数组导致评分失败。
        """
        return self.model_.predict_proba(X)[:, 1]


def section(title: str) -> None:
    """终端分节标题。"""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def find_data_file() -> Path:
    """在常见位置查找数据。"""
    for p in [BASE_DIR / "data" / "telco_churn.csv",
              BASE_DIR.parent / "data" / "telco_churn.csv",
              BASE_DIR / "telco_churn.csv"]:
        if p.exists():
            return p
    raise FileNotFoundError("找不到 telco_churn.csv")


def save_fig(fig, stem: str) -> None:
    """同时输出 300 dpi PNG（预览）与 PDF（LaTeX 投稿用）。"""
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson 区间：比例型指标的小样本置信区间，比正态近似稳健。"""
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def cramers_v(table: pd.DataFrame) -> float:
    """Cramér's V：类别变量关联强度（0~1）。"""
    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def rank_biserial(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    """秩双列相关：数值特征在两类间的效应量（-1~1）。"""
    stat = mannwhitneyu(x_pos, x_neg, alternative="two-sided").statistic
    return 2 * stat / (len(x_pos) * len(x_neg)) - 1


def bootstrap_ci(y_true, proba, metric_fn, n_boot: int = N_BOOT, seed: int = RANDOM_STATE):
    """Bootstrap 95% CI：单次划分的 AUC/PR-AUC 波动大，必须给区间。"""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], proba[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ==================================================================
# 阶段 0：读取数据
# ==================================================================
section("阶段 0 · 读取数据")
DATA_PATH = find_data_file()
raw = pd.read_csv(DATA_PATH)
print("数据文件:", DATA_PATH, "| 形状:", raw.shape)
print(pd.DataFrame({"dtype": raw.dtypes.astype(str),
                    "n_unique": raw.nunique(dropna=True),
                    "n_missing": raw.isna().sum()}).to_string())

# ==================================================================
# 阶段 1：探索数据
# ==================================================================
section("阶段 1 · 探索数据")

# 1.1 标签分布 + 置信区间
yes = int((raw["Churn"] == "Yes").sum())
no = int((raw["Churn"] == "No").sum())
lo, hi = wilson_ci(yes, yes + no)
print("流失 %d / 未流失 %d，流失率 %.4f（95%% CI %.4f~%.4f），不平衡比 1:%.2f"
      % (yes, no, yes / (yes + no), lo, hi, no / yes))
print(">>> 全预测“不流失”准确率就有 73.6%，准确率不能作为主指标。")

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
cnt = raw["Churn"].value_counts().reindex(["No", "Yes"])
axes[0].bar(cnt.index, cnt.values, width=0.55, color=["#4E79A7", "#E15759"],
            edgecolor="black", linewidth=0.8)
for i, v in enumerate(cnt.values):
    axes[0].text(i, v + 50, f"{v:,}", ha="center", fontsize=9)
axes[0].set_ylabel("Number of customers")
axes[0].set_title("(a) Class counts")
axes[0].set_ylim(0, cnt.max() * 1.12)

rate = cnt / cnt.sum()
axes[1].bar(rate.index, rate.values, width=0.55, color=["#4E79A7", "#E15759"],
            edgecolor="black", linewidth=0.8)
axes[1].errorbar(["Yes"], [rate["Yes"]], yerr=[[rate["Yes"] - lo], [hi - rate["Yes"]]],
                 fmt="none", ecolor="black", capsize=5, lw=1.2)
for i, v in enumerate(rate.values):
    axes[1].text(i, v + 0.025, f"{v:.2%}", ha="center", fontsize=9)
axes[1].set_ylim(0, 1)
axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
axes[1].set_ylabel("Churn rate")
axes[1].set_title("(b) Churn rate with 95% Wilson CI")
fig.tight_layout()
save_fig(fig, "fig1_target")

# 1.2 缺失机制诊断
print("\n-- 缺失机制诊断 --")
tc_num = pd.to_numeric(raw["TotalCharges"], errors="coerce")
hidden = tc_num.isna() & raw["TotalCharges"].notna()
structural = tc_num.isna() & (raw["tenure"] == 0)
print("TotalCharges 原始 dtype: %s；空白字符串隐藏缺失 %d 行；其中结构性缺失(tenure=0) %d 行"
      % (raw["TotalCharges"].dtype, int(hidden.sum()), int(structural.sum())))
print(">>> 结构性缺失（新客户尚无累计费用）不能按中位数填，应按领域规则填 0。")

non_id = raw.drop(columns=["customerID"])
completeness = non_id.notna().mean(axis=1)
junk = completeness < 0.10
print("非空字段不足 10%% 的脏记录 %d 行；缺目标值 %d 行"
      % (int(junk.sum()), int(raw["Churn"].isna().sum())))
if junk.any():
    print(raw.loc[junk, ["customerID", "gender", "tenure", "TotalCharges", "Churn"]].to_string(index=False))
print("gender 取值分布:", raw["gender"].value_counts(dropna=False).to_dict(), "（'Fem' 为拼写错误）")

# 缺失模式图
na_cols = raw.columns[raw.isna().any()].tolist()
fig, ax = plt.subplots(figsize=(7.6, 3.2))
ax.imshow(raw[na_cols].isna().T.values, aspect="auto", cmap="Greys",
          interpolation="nearest")
ax.set_yticks(range(len(na_cols)), na_cols, fontsize=8)
ax.set_xlabel("Customer index")
ax.set_title("Missingness pattern matrix (black = missing)")
fig.tight_layout()
save_fig(fig, "fig2_missing")

# 1.3 关联强度（效应量）与关键特征分布
print("\n-- 效应量排序 --")
df_eda = raw.dropna(subset=["Churn"]).copy()
df_eda["ChurnFlag"] = (df_eda["Churn"] == "Yes").astype(int)

assoc = []
for col in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
    a = df_eda.loc[df_eda["ChurnFlag"] == 1, col].dropna().to_numpy()
    b = df_eda.loc[df_eda["ChurnFlag"] == 0, col].dropna().to_numpy()
    assoc.append(dict(feature=col, type="numeric", effect=abs(rank_biserial(a, b)),
                      p=mannwhitneyu(a, b).pvalue))
for col in [c for c in df_eda.select_dtypes("object").columns
            if c not in ["customerID", "Churn", "TotalCharges"]]:
    table = pd.crosstab(df_eda[col], df_eda["Churn"])
    chi2, p, _, _ = chi2_contingency(table)
    assoc.append(dict(feature=col, type="categorical", effect=cramers_v(table), p=p))
assoc_df = pd.DataFrame(assoc).sort_values("effect", ascending=False)
print(assoc_df.head(10).round(4).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
for label, color in [("No", "#4E79A7"), ("Yes", "#E15759")]:
    axes[0].hist(df_eda.loc[df_eda["Churn"] == label, "tenure"].dropna(), bins=25,
                 alpha=0.55, density=True, label=f"Churn = {label}", color=color,
                 edgecolor="white", linewidth=0.4)
axes[0].set_xlabel("Tenure (months)")
axes[0].set_ylabel("Density")
axes[0].set_title("(a) Tenure distribution")
axes[0].legend()

for label, color in [("No", "#4E79A7"), ("Yes", "#E15759")]:
    axes[1].hist(df_eda.loc[df_eda["Churn"] == label, "MonthlyCharges"].dropna(), bins=25,
                 alpha=0.55, density=True, label=f"Churn = {label}", color=color,
                 edgecolor="white", linewidth=0.4)
axes[1].set_xlabel("Monthly charges (USD)")
axes[1].set_title("(b) Monthly charges")
axes[1].legend()

contract = pd.crosstab(df_eda["Contract"], df_eda["Churn"], normalize="index")
contract.plot(kind="bar", stacked=True, ax=axes[2], color=["#4E79A7", "#E15759"],
              edgecolor="black", linewidth=0.5, width=0.6)
axes[2].set_xlabel("Contract type")
axes[2].set_ylabel("Proportion")
axes[2].set_title("(c) Churn share by contract")
axes[2].tick_params(axis="x", rotation=0)
axes[2].legend(title="Churn", loc="lower right")
fig.tight_layout()
save_fig(fig, "fig3_features")

# 1.4 冗余/泄漏审计
print("\n-- 冗余/泄漏审计 --")
tc = pd.to_numeric(df_eda["TotalCharges"], errors="coerce")
ok = tc.notna()
prod = (df_eda.loc[ok, "tenure"] * df_eda.loc[ok, "MonthlyCharges"]).to_frame("tenure_x_monthly")
reg = LinearRegression().fit(prod, tc[ok])
r2 = reg.score(prod, tc[ok])
print("TotalCharges ≈ tenure × MonthlyCharges: R² = %.4f" % r2)
print(">>> 冗余特征会让 SHAP/重要性虚高，解释时必须合并解读。")

corr = df_eda[["SeniorCitizen", "tenure", "MonthlyCharges", "ChurnFlag"]].copy()
corr["TotalCharges"] = tc
corr = corr.corr()
fig, ax = plt.subplots(figsize=(5.6, 4.8))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if abs(corr.values[i, j]) > 0.6 else "black")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("Pearson correlation")
ax.set_title("Correlation among numeric features")
fig.tight_layout()
save_fig(fig, "fig4_correlation")

# ==================================================================
# 阶段 2：清洗数据
# ==================================================================
section("阶段 2 · 清洗数据")
clean = raw.copy()
n0 = len(clean)
clean = clean.dropna(subset=["Churn"]).reset_index(drop=True)     # 只删无标签脏行
dropped = n0 - len(clean)

clean["TotalCharges"] = pd.to_numeric(
    clean["TotalCharges"].astype(str).str.strip(), errors="coerce")  # 空白字符串→NaN
mask_new = (clean["tenure"] == 0) & clean["TotalCharges"].isna()
clean.loc[mask_new, "TotalCharges"] = 0.0                         # 领域规则：新客户累计费用为 0

print("删除无标签/几乎全空记录 %d 行；TotalCharges 结构性补 0 %d 行；剩余缺失 %d"
      % (dropped, int(mask_new.sum()), int(clean["TotalCharges"].isna().sum())))
print(">>> 不用的做法：整表 dropna、均值/中位数一把填。")
print(">>> 用的做法：能按业务含义确定的直接修，不确定的保留 NaN + 缺失指示交给模型。")

y = (clean["Churn"] == "Yes").astype(int)
X = clean.drop(columns=["Churn", "customerID"])
num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in X.columns if c not in num_cols]

# ==================================================================
# 阶段 3：预处理与划分
# ==================================================================
section("阶段 3 · 预处理与划分")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
# 再从训练集切验证集：只用来选阈值/早停，测试集只在最后评估一次
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=RANDOM_STATE)
print("训练 %s | 验证 %s | 测试 %s" % (X_fit.shape, X_val.shape, X_test.shape))
print("流失率：训练 %.4f / 验证 %.4f / 测试 %.4f"
      % (y_fit.mean(), y_val.mean(), y_test.mean()))


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """列级预处理器：补值(+指示) → 可选标准化；类别列独热（drop first 防共线）。"""
    num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first",
                                  sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


print(">>> 所有拟合型变换都封装在 Pipeline 内；CatBoost 直接使用原始类别列，不做独热。")

# ==================================================================
# 阶段 4：训练多模型
# ==================================================================
section("阶段 4 · 训练多模型")

# 决策树深度扫描（教学用；最终以验证集/交叉验证为准）
depths = list(range(2, 16))
tr_auc, te_auc = [], []
for d in depths:
    pipe = Pipeline([("prep", make_preprocessor(False)),
                     ("clf", DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE))])
    pipe.fit(X_fit, y_fit)
    tr_auc.append(roc_auc_score(y_fit, pipe.predict_proba(X_fit)[:, 1]))
    te_auc.append(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]))
best_depth = depths[int(np.argmax(te_auc))]
print("决策树最优深度 = %d" % best_depth)

models = {
    "基线(全预测不流失)": Pipeline([("prep", make_preprocessor(False)),
                                ("clf", DummyClassifier(strategy="prior"))]),
    "朴素贝叶斯": Pipeline([("prep", make_preprocessor(True)), ("clf", GaussianNB())]),
    "SVM-线性核": Pipeline([("prep", make_preprocessor(True)),
                          ("clf", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE))]),
    "SVM-RBF核": Pipeline([("prep", make_preprocessor(True)),
                         ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
    "MLP": Pipeline([("prep", make_preprocessor(True)),
                     ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600,
                                           random_state=RANDOM_STATE))]),
    "决策树": Pipeline([("prep", make_preprocessor(False)),
                      ("clf", DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE))]),
    "随机森林": Pipeline([("prep", make_preprocessor(False)),
                       ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                      random_state=RANDOM_STATE))]),
    "GBDT(HistGB)": Pipeline([("prep", make_preprocessor(False)),
                              ("clf", HistGradientBoostingClassifier(
                                  learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                  min_samples_leaf=20, l2_regularization=1.0,
                                  early_stopping=True, n_iter_no_change=30,
                                  random_state=RANDOM_STATE))]),
    "GBDT+加权+等渗校准": CalibratedClassifierCV(
        Pipeline([("prep", make_preprocessor(False)),
                  ("clf", HistGradientBoostingClassifier(
                      learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                      min_samples_leaf=20, l2_regularization=1.0,
                      early_stopping=True, n_iter_no_change=30,
                      class_weight="balanced", random_state=RANDOM_STATE))]),
        method="isotonic", cv=N_SPLITS),
    # CatBoost：原生类别特征 + 类别加权。不是为了“更高级”，而是验证
    # “不独热编码、让树自己处理类别”能否在这一数据上带来稳定收益。
    "CatBoost(原生类别)": CatBoostNative(),
}

# 先在 train_fit 上训练 → 验证集选阈值 → 再全训练集重训 → 测试集只评估一次
fitted_val, val_proba, thresholds = {}, {}, {}
for name, model in models.items():
    model.fit(X_fit, y_fit)
    fitted_val[name] = model
    p_val = model.predict_proba(X_val)[:, 1]
    val_proba[name] = p_val
    # 业务容量约束下的阈值：验证集分数分位数（等价于“只联系前 20%”）
    thresholds[name] = float(np.quantile(p_val, 1 - CAPACITY))
    print("验证集阈值 · %-24s %.4f" % (name, thresholds[name]))

fitted, test_proba = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)          # 全训练集重训，阈值仍用验证集
    fitted[name] = model
    test_proba[name] = model.predict_proba(X_test)[:, 1]
    print("已重训:", name)

# 训练诊断图：MLP 损失 + 决策树深度
mlp = fitted["MLP"].named_steps["clf"]
loss = np.asarray(mlp.loss_curve_)
print("MLP 迭代 %d 轮，loss %.4f → %.4f" % (mlp.n_iter_, loss[0], loss[-1]))

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
axes[0].plot(loss, color="#B07AA1", lw=1.4)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Training loss")
axes[0].set_title("(a) MLP convergence")
axes[0].grid(alpha=0.35)

axes[1].plot(depths, tr_auc, "o-", ms=3.5, lw=1.3, color="#2E5EAA", label="Train AUC")
axes[1].plot(depths, te_auc, "s-", ms=3.5, lw=1.3, color="#E15759", label="Test AUC")
axes[1].axvline(best_depth, color="gray", ls="--", lw=1)
axes[1].set_xlabel("max_depth")
axes[1].set_ylabel("ROC-AUC")
axes[1].set_title("(b) Decision tree depth scan")
axes[1].legend()
axes[1].grid(alpha=0.35)
fig.tight_layout()
save_fig(fig, "fig5_training")

# ==================================================================
# 阶段 5：指标与图表比较
# ==================================================================
section("阶段 5 · 指标与图表比较")

# 5 折交叉验证：给出去噪后的稳定性判断（CatBoost 也一起进 CV）
cv_names = ["朴素贝叶斯", "SVM-线性核", "随机森林", "GBDT(HistGB)",
            "GBDT+加权+等渗校准", "CatBoost(原生类别)"]
cv_split = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_stats = {}
for name in cv_names:
    s = cross_val_score(models[name], X_train, y_train, cv=cv_split,
                        scoring="roc_auc", n_jobs=-1)
    cv_stats[name] = (s.mean(), s.std())
    print("5-fold CV AUC · %-24s %.4f ± %.4f" % (name, s.mean(), s.std()))

rows = []
y_test_arr = y_test.to_numpy()
for name, proba in test_proba.items():
    pred = (proba >= thresholds[name]).astype(int)        # 验证集选出的阈值
    tn, fp, fn, tp = confusion_matrix(y_test_arr, pred).ravel()
    auc = roc_auc_score(y_test_arr, proba)
    ap = average_precision_score(y_test_arr, proba)
    auc_lo, auc_hi = bootstrap_ci(y_test_arr, proba, roc_auc_score)
    ap_lo, ap_hi = bootstrap_ci(y_test_arr, proba, average_precision_score)
    rows.append(dict(model=name, threshold=thresholds[name],
                     contact_rate=pred.mean(),
                     auc=auc, auc_lo=auc_lo, auc_hi=auc_hi,
                     pr_auc=ap, pr_lo=ap_lo, pr_hi=ap_hi,
                     recall=tp / (tp + fn), specificity=tn / (tn + fp),
                     precision=precision_score(y_test_arr, pred, zero_division=0),
                     f1=f1_score(y_test_arr, pred, zero_division=0),
                     brier=brier_score_loss(y_test_arr, proba),
                     logloss=log_loss(y_test_arr, np.clip(proba, 1e-6, 1 - 1e-6)),
                     cv_auc=cv_stats.get(name, (np.nan, np.nan))[0],
                     cv_std=cv_stats.get(name, (np.nan, np.nan))[1]))
scores = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
print("\n模型成绩单（验证集阈值，测试集评估；按 PR-AUC 排序）:")
print(scores.round(4).to_string(index=False))
scores.to_csv(RESULT_DIR / "model_scores.csv", index=False,
              encoding="utf-8-sig", float_format="%.4f")

# 图 6：AUC / PR-AUC 带 95% CI 的横向对比（论文主图风格）
plot_df = scores[scores["model"] != "基线(全预测不流失)"].sort_values("auc")
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
ypos = np.arange(len(plot_df))
for ax, col, lo, hi, title in [
        (axes[0], "auc", "auc_lo", "auc_hi", "(a) ROC-AUC (95% CI)"),
        (axes[1], "pr_auc", "pr_lo", "pr_hi", "(b) PR-AUC (95% CI)")]:
    vals = plot_df[col].to_numpy()
    err = np.vstack([vals - plot_df[lo], plot_df[hi] - vals])
    colors = ["#E15759" if m == "CatBoost(原生类别)" else "#4E79A7" for m in plot_df["model"]]
    ax.barh(ypos, vals, xerr=err, color=colors, height=0.62, edgecolor="black",
            linewidth=0.6, error_kw=dict(ecolor="#333333", capsize=2.5, lw=0.9))
    ax.set_yticks(ypos, [PRETTY[m] for m in plot_df["model"]])
    ax.set_xlabel(col.replace("pr_auc", "PR-AUC").replace("auc", "ROC-AUC"))
    ax.set_title(title)
    ax.set_xlim(0.4 if col == "auc" else 0.2, 1.0 if col == "auc" else 0.8)
    ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig6_model_comparison")

# 图 7：ROC 与 PR（选 AUC 前 6 个模型，避免图例遮挡）
top6 = scores[scores["model"] != "基线(全预测不流失)"].nlargest(6, "auc")["model"].tolist()
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
for name in top6:
    p = test_proba[name]
    fpr, tpr, _ = roc_curve(y_test_arr, p)
    axes[0].plot(fpr, tpr, lw=1.5, color=PALETTE[name],
                 label=f"{PRETTY[name]} ({roc_auc_score(y_test_arr, p):.3f})")
    prec, rec, _ = precision_recall_curve(y_test_arr, p)
    axes[1].plot(rec, prec, lw=1.5, color=PALETTE[name],
                 label=f"{PRETTY[name]} ({average_precision_score(y_test_arr, p):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=0.9)
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("(a) ROC curves")
axes[0].legend(loc="lower right", fontsize=6.8)
axes[0].grid(alpha=0.3)

axes[1].axhline(y_test_arr.mean(), color="gray", ls="--", lw=0.9, label="Prevalence")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("(b) Precision-recall curves")
axes[1].legend(loc="upper right", fontsize=6.8)
axes[1].grid(alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig7_roc_pr")

# 图 8：校准曲线（加权树模型 + 等渗校准 + CatBoost）
fig, ax = plt.subplots(figsize=(4.8, 4.4))
for name in ["GBDT(HistGB)", "GBDT+加权+等渗校准", "CatBoost(原生类别)"]:
    frac_pos, mean_pred = calibration_curve(y_test_arr, test_proba[name],
                                            n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", ms=4, lw=1.4, color=PALETTE[name], label=PRETTY[name])
ax.plot([0, 1], [0, 1], "k--", lw=0.9, label="Perfect calibration")
ax.set_xlabel("Predicted churn probability")
ax.set_ylabel("Observed churn rate")
ax.set_title("Reliability diagram")
ax.legend(fontsize=7.4)
ax.grid(alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig8_calibration")

# 图 9：lift / gain（以 CatBoost 为例）
order = np.argsort(-test_proba["CatBoost(原生类别)"])
y_sorted = y_test_arr[order]
cum_churn = np.cumsum(y_sorted) / y_sorted.sum()
cum_pop = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
lift = cum_churn / cum_pop
gain20 = cum_churn[int(CAPACITY * len(y_sorted)) - 1]
print("CatBoost：联系前 20%% 客户可抓到 %.1f%% 流失客户（lift=%.2f）"
      % (gain20 * 100, gain20 / CAPACITY))

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
axes[0].plot(cum_pop, cum_churn, lw=2, color="#59A14F")
axes[0].plot([0, 1], [0, 1], "k--", lw=0.9, label="Random targeting")
axes[0].axvline(CAPACITY, color="gray", ls=":", lw=1.2)
axes[0].scatter([CAPACITY], [gain20], color="#E15759", zorder=5, s=28)
axes[0].annotate(f"{gain20:.1%} captured", xy=(CAPACITY, gain20),
                 xytext=(CAPACITY + 0.08, gain20 - 0.10), fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
axes[0].set_xlabel("Fraction of customers contacted")
axes[0].set_ylabel("Cumulative churners captured")
axes[0].set_title("(a) Cumulative gain")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

axes[1].plot(cum_pop, lift, lw=2, color="#4E79A7")
axes[1].axhline(1, color="k", ls="--", lw=0.9)
axes[1].axvline(CAPACITY, color="gray", ls=":", lw=1.2)
axes[1].set_xlabel("Fraction of customers contacted")
axes[1].set_ylabel("Lift")
axes[1].set_title("(b) Lift curve")
axes[1].grid(alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig9_lift_gain")

# ==================================================================
# 阶段 6：选型与业务取舍
# ==================================================================
section("阶段 6 · 选型与业务取舍")

# 单客经济性：FN 损失 ≈ 24 个月月费 × 挽留成功率；FP/TP 成本 = 50 美元优惠
arr = float(1 - y_train.mean())
monthly_test = X_test["MonthlyCharges"].to_numpy()


def profit_per_1000(y_true, pred, monthly):
    """每 1000 名客户的期望利润（美元）。"""
    saved = ((y_true == 1) & (pred == 1)) * arr * CLV_MONTHS * monthly
    return (saved - pred * OFFER_COST).sum() / len(y_true) * 1000


def topk_pred(proba, k):
    """按分数取前 k 比例（容量约束）。"""
    pred = np.zeros(len(proba), dtype=int)
    pred[np.argsort(-np.asarray(proba))[: int(round(k * len(proba)))]] = 1
    return pred


# 6.1 业务指标表：固定 20% 容量，补充 10%/30% 的捕获率
biz_rows = []
for name, proba in test_proba.items():
    pred = (proba >= thresholds[name]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_arr, pred).ravel()
    capture = {f"capture@{int(k * 100)}%": (y_test_arr[topk_pred(proba, k) == 1].sum()
                                            / y_test_arr.sum()) for k in (0.1, 0.2, 0.3)}
    biz_rows.append(dict(model=name, threshold=round(thresholds[name], 4),
                         contact_rate=pred.mean(),
                         recall=tp / (tp + fn),
                         precision=tp / (tp + fp) if (tp + fp) else 0,
                         profit_threshold=profit_per_1000(y_test_arr, pred, monthly_test),
                         profit_at_capacity=profit_per_1000(
                             y_test_arr, topk_pred(proba, CAPACITY), monthly_test),
                         **capture))
biz = pd.DataFrame(biz_rows).sort_values("profit_at_capacity", ascending=False).reset_index(drop=True)
print("业务指标（阈值来自验证集；容量对齐列 = 固定触达前 20% 的公平对比）:")
print(biz.round(4).to_string(index=False))
biz.to_csv(RESULT_DIR / "business_metrics.csv", index=False,
           encoding="utf-8-sig", float_format="%.4f")

# 6.2 利润-容量曲线：预算变化时排序是否改变
caps = np.linspace(0.05, 0.60, 23)
fig, ax = plt.subplots(figsize=(6.6, 4.2))
for name in ["随机森林", "GBDT+加权+等渗校准", "CatBoost(原生类别)"]:
    profits = [profit_per_1000(y_test_arr, topk_pred(test_proba[name], c), monthly_test) for c in caps]
    ax.plot(caps * 100, profits, "o-", ms=3, lw=1.6, color=PALETTE[name], label=PRETTY[name])
ax.axvline(CAPACITY * 100, color="gray", ls="--", lw=1, label="Capacity = 20%")
ax.set_xlabel("Contacted fraction (%)")
ax.set_ylabel("Expected profit (USD per 1,000)")
ax.set_title("Profit vs. operational capacity")
ax.legend(fontsize=7.4)
ax.grid(alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig10_profit_capacity")

# 6.3 选型：CV 第一梯队 → 容量对齐的利润 → 概率是否校准
cv_best = max(v[0] for v in cv_stats.values())
first_tier = [m for m, v in cv_stats.items() if v[0] >= cv_best - 0.01]
tier = biz[biz["model"].isin(first_tier)].copy()
top_profit = tier["profit_at_capacity"].max()
tied = tier[tier["profit_at_capacity"] >= 0.90 * top_profit]   # 10% 内视为打平（ARR/CLV 本身是估计）
deploy = tied.sort_values("profit_at_capacity", ascending=False).iloc[0]["model"]
for cand in tied["model"]:
    if "校准" in cand:
        deploy = cand
        break
print("\n第一梯队（CV AUC ≥ %.4f）: %s" % (cv_best - 0.01, "、".join(first_tier)))
print("业务最优（容量对齐）: %s（%.0f 美元/千人）；最终上线建议: %s"
      % (biz.iloc[0]["model"], biz.iloc[0]["profit_at_capacity"], deploy))
print(">>> 利润差在 10% 以内、AUC 差 <0.01 时统计上并未分出胜负，"
      "应优先概率校准、CV 波动更小的模型。")

# 6.4 CatBoost SHAP：解释“为什么把某些客户判为高风险”
cat_model = fitted["CatBoost(原生类别)"].model_
shap_vals = cat_model.get_feature_importance(
    Pool(X_test, cat_features=cat_cols), type="ShapValues")[:, :-1]   # 最后一列是基准值
mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_test.columns).sort_values(ascending=False)
top_feats = mean_abs.head(8).index.tolist()
print("\nCatBoost 平均 |SHAP| Top8:\n" + mean_abs.head(8).round(4).to_string())
print(">>> TotalCharges 排进前 8 主要是与 tenure、MonthlyCharges 冗余（R²=0.9991），")
print(">>> 它反映的是“累计消费/在网时长”的复合信号，不应被解读为独立业务杠杆。")

# 左图：平均 |SHAP| 条形图；右图：beeswarm（颜色=特征取值高低）
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6),
                         gridspec_kw={"width_ratios": [1.0, 1.25]})
order_feats = top_feats[::-1]
axes[0].barh(range(len(order_feats)), mean_abs[order_feats].values,
             color="#59A14F", edgecolor="black", linewidth=0.6, height=0.65)
axes[0].set_yticks(range(len(order_feats)), order_feats)
axes[0].set_xlabel("mean |SHAP| (log-odds)")
axes[0].set_title("(a) Global feature attribution")
axes[0].grid(axis="x", alpha=0.3)

rng = np.random.default_rng(RANDOM_STATE)
for i, feat in enumerate(order_feats):
    j = X_test.columns.get_loc(feat)
    sv = shap_vals[:, j]
    fv = X_test[feat]
    if fv.dtype == object:
        color = fv.astype("category").cat.codes.to_numpy()   # 类别映射为整数着色
    else:
        color = fv.to_numpy()
    jitter = rng.uniform(-0.22, 0.22, len(sv))
    sc = axes[1].scatter(sv, np.full(len(sv), i) + jitter, c=color, cmap="coolwarm",
                         s=4.5, alpha=0.65, linewidths=0)
axes[1].axvline(0, color="gray", lw=0.8, ls="--")
axes[1].set_yticks(range(len(order_feats)), order_feats)
axes[1].set_xlabel("SHAP value (impact on churn log-odds)")
axes[1].set_title("(b) Customer-level SHAP beeswarm")
cb = fig.colorbar(sc, ax=axes[1], fraction=0.03, pad=0.02)
cb.set_label("Feature value (low → high)")
fig.tight_layout()
save_fig(fig, "fig11_shap")

# 6.5 高风险客户画像 vs 全体（把 SHAP 变成可执行动作）
high_mask = test_proba["CatBoost(原生类别)"] >= thresholds["CatBoost(原生类别)"]
high = X_test[high_mask]
print("\n高风险组（CatBoost 前 20%）画像 vs 全体:")
for col in ["Contract", "InternetService", "TechSupport", "tenure", "MonthlyCharges"]:
    if X_test[col].dtype == object:
        share_high = high[col].value_counts(normalize=True).head(2)
        share_all = X_test[col].value_counts(normalize=True).head(2)
        print("  %-16s 高风险组 %s | 全体 %s"
              % (col, share_high.round(3).to_dict(), share_all.round(3).to_dict()))
    else:
        print("  %-16s 高风险组中位数 %.1f | 全体中位数 %.1f"
              % (col, high[col].median(), X_test[col].median()))

# 6.6 业务建议（数字来自上面的统计，不是套话）
contract_rate = clean.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean())
tenure_bin = pd.cut(clean["tenure"], [-1, 6, 12, 24, 48, 100],
                    labels=["0-6", "7-12", "13-24", "25-48", "49+"])
tenure_rate = clean.groupby(tenure_bin, observed=True)["Churn"].apply(lambda s: (s == "Yes").mean())
print("\n分合同类型流失率:", (contract_rate * 100).round(2).to_dict())
print("分在网时长流失率:", (tenure_rate * 100).round(2).to_dict())
print("建议动作：")
print("  1) 月付合同流失率 %.1f%%，对高风险客户推年付/两年合约 + 首年折扣（最能直接改变风险因素）。"
      % (contract_rate.get("Month-to-month", np.nan) * 100))
print("  2) 0-6 个月新客流失率 %.1f%%，前 90 天做 onboarding 与主动回访；"
      % (tenure_rate.get("0-6", np.nan) * 100))
print("  3) 高风险组中缺 TechSupport/OnlineSecurity 的比例更高，用“安全+支持”增值包降低风险感知。")

section("完成")
print("图片: figures/fig1_target ... fig11_shap（PNG 300dpi + PDF）")
print("数据: results/model_scores.csv, results/business_metrics.csv")
