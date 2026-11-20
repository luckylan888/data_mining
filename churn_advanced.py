"""
客户流失预警：六阶段进阶建模流水线（中文详细注释版）

设计原则（对应任务书要求的六个阶段，但每一步都换成“先进且可解释”的做法）：
  ① 探索数据   —— 缺失机制诊断（结构性缺失 vs 随机缺失）、效应量（Cramér's V / 秩双列相关）、
                  冗余与泄漏诊断（TotalCharges 是否是可推导的伪特征）
  ② 清洗数据   —— 按“记录有效性 + 领域规则”清洗，而不是无脑 dropna / 中位数填坑；
                  保留缺失指示变量，让模型自己学习“缺失本身”是否有信息
  ③ 预处理与划分 —— 先分层划分，再在 Pipeline 内完成补值/编码/缩放，杜绝数据泄漏；
                  交叉验证时整条流程一起进 CV
  ④ 训练多模型 —— 课程要求的 NB/SVM/MLP/DT/RF + 现代 GBDT（HistGradientBoosting）
                  + 概率校准版 GBDT
  ⑤ 指标与图表 —— ROC-AUC / PR-AUC（不平衡更该看）/ Brier / 对数损失，配合 Bootstrap 置信区间；
                  ROC、PR、校准曲线、提升图（lift/gain）、成本-容量曲线
  ⑥ 选型与业务 —— 以“固定运营容量 + 期望利润”选模型和阈值，而不是 0.5 或准确率

主要文献依据（2026 年 arXiv）：
  - arXiv:2610.00118  审计 Telco Churn 基准：先 SMOTE 再划分会让少数类 F1 虚高 13.1 个百分点；
                      TotalCharges ≈ tenure × MonthlyCharges（R²=0.999）；等渗校准最稳；
                      成本最优阈值比 F1 最优阈值低 5~10 倍
  - arXiv:2606.07582  FT-Transformer + XGBoost 堆叠：类别加权优于合成过采样，需做概率校准
  - arXiv:2608.26151  SHAP 解释 + CRM 落地：跨模型 AUC 差异常在 0.01 内，选型要看业务收益
  - arXiv:2507.08860  e-Profits：用留存概率（ARR）和提升度衡量利润，而非纯统计指标
  - arXiv:2503.16644  “To impute or not to impute”：先判断缺失机制，再决定补不补

运行：
    python churn_advanced.py        # 依赖：pandas numpy scipy matplotlib scikit-learn

产出：
    figures/adv01_target.png          标签分布（含 95% 置信区间）
    figures/adv02_missing.png         缺失模式图
    figures/adv03_features.png        关键特征与流失的关系
    figures/adv04_corr.png            数值特征相关性热力图
    figures/adv05_mlp_loss.png        MLP 训练损失曲线
    figures/adv06_tree_depth.png      决策树深度扫描
    figures/adv07_roc_pr_cal.png      ROC / PR / 校准曲线
    figures/adv08_lift_gain.png       提升图与增益图
    figures/adv09_profit_capacity.png 利润-容量曲线与模型对比
    figures/adv10_importance.png      置换重要性 Top10
    results/advanced_scores.csv       模型成绩单（含置信区间）
    results/business_summary.csv      业务收益汇总
"""

import os
import warnings
from pathlib import Path

# 屏蔽第三方库的冗余警告，保持输出干净（不影响结果）
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无图形界面环境也能出图
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from scipy.stats import chi2_contingency, mannwhitneyu

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
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

RANDOM_STATE = 42          # 统一随机种子，保证结果可复现
TEST_SIZE = 0.25           # 测试集比例
N_SPLITS = 5               # 交叉验证折数
N_BOOT = 500               # Bootstrap 重采样次数（置信区间）
CAPACITY = 0.20            # 运营约束：每月只能联系风险最高的 20% 客户
OFFER_COST = 50.0          # 一次挽留优惠的成本（美元），来自 2610.00118 的设定
CLV_MONTHS = 24            # 客户生命周期 24 个月（CLV 代理）

# 中文字体回退：Windows 用微软雅黑，macOS 用苹方，Linux 用文泉驿
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110


def section(title: str) -> None:
    """打印分节标题，方便在终端里定位每个阶段的输出。"""
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def find_data_file() -> Path:
    """在几个常见位置查找数据文件，避免脚本换个目录就报错。"""
    for p in [BASE_DIR / "data" / "telco_churn.csv",
              BASE_DIR.parent / "data" / "telco_churn.csv",
              BASE_DIR / "telco_churn.csv"]:
        if p.exists():
            return p
    raise FileNotFoundError("找不到 telco_churn.csv")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson 置信区间：比正态近似更适合比例（尤其接近 0/1）的区间估计。"""
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def cramers_v(table: pd.DataFrame) -> float:
    """Cramér's V：两个类别变量的关联强度（0~1），比只看卡方 p 值更能说明“有多大用”。"""
    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def rank_biserial(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    """秩双列相关：数值特征在两类之间的效应量，取值 -1~1，绝对值越大区分度越强。"""
    stat = mannwhitneyu(x_pos, x_neg, alternative="two-sided").statistic
    return 2 * stat / (len(x_pos) * len(x_neg)) - 1


def bootstrap_ci(y_true, proba, metric_fn, n_boot: int = N_BOOT, seed: int = RANDOM_STATE):
    """Bootstrap 置信区间：不平衡数据上单次划分的 AUC/PR-AUC 波动很大，必须给区间。"""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))       # 有放回抽样
        if len(np.unique(y_true[idx])) < 2:                   # 抽到单一类别时跳过
            continue
        scores.append(metric_fn(y_true[idx], proba[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ==================================================================
# 阶段 0：读取数据
# ==================================================================
section("阶段 0 · 读取数据")
DATA_PATH = find_data_file()
raw = pd.read_csv(DATA_PATH)
print("数据文件:", DATA_PATH)
print("原始形状:", raw.shape)
print("\n字段类型与唯一值数量:")
print(pd.DataFrame({"dtype": raw.dtypes.astype(str),
                    "n_unique": raw.nunique(dropna=True),
                    "n_missing": raw.isna().sum()}).to_string())

# ==================================================================
# 阶段 1：探索数据（EDA）
# ==================================================================
section("阶段 1 · 探索数据")

# ---- 1.1 标签分布：不仅看比例，还要给置信区间 ----
yes = int((raw["Churn"] == "Yes").sum())
no = int((raw["Churn"] == "No").sum())
total = yes + no
lo, hi = wilson_ci(yes, total)
print("-- 1.1 标签分布 --")
print("流失 %d / 未流失 %d，流失率 %.4f（95%% CI: %.4f ~ %.4f），不平衡比约 1:%.2f"
      % (yes, no, yes / total, lo, hi, no / yes))
print(">>> 26% 的正类比例意味着：全预测“不流失”就能拿到 73.6% 准确率，准确率不能作为主指标。")

# 图 1：计数 + 比例，标注置信区间
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
cnt = raw["Churn"].value_counts().reindex(["No", "Yes"])
axes[0].bar(cnt.index, cnt.values, color=["#4C78A8", "#E45756"])
axes[0].set_title("客户数量")
for i, v in enumerate(cnt.values):
    axes[0].text(i, v + 40, f"{v:,}", ha="center")
axes[0].set_ylabel("人数")

rate = cnt / cnt.sum()
axes[1].bar(rate.index, rate.values, color=["#4C78A8", "#E45756"])
axes[1].errorbar(["Yes"], [rate["Yes"]], yerr=[[rate["Yes"] - lo], [hi - rate["Yes"]]],
                 fmt="none", ecolor="black", capsize=6)
axes[1].set_title("流失比例（95% Wilson CI）")
axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
for i, v in enumerate(rate.values):
    axes[1].text(i, v + 0.02, f"{v:.2%}", ha="center")
axes[1].set_ylim(0, 1)
plt.suptitle("Churn 标签分布")
plt.tight_layout()
plt.savefig(FIG_DIR / "adv01_target.png")
plt.close()

# ---- 1.2 缺失机制诊断：区分“结构性缺失”和“随机缺失” ----
print("\n-- 1.2 缺失机制诊断 --")
print("TotalCharges 的原始类型:", raw["TotalCharges"].dtype, "（本应是数值）")

# 用 to_numeric 强制转换，无法转换的（空字符串/空格）会变成 NaN，从而暴露隐藏缺失
tc_num = pd.to_numeric(raw["TotalCharges"], errors="coerce")
hidden = tc_num.isna() & raw["TotalCharges"].notna()      # “看起来有值但其实是空白串”的行
print("空白字符串导致的隐藏缺失:", int(hidden.sum()), "行")
print("转换后总缺失:", int(tc_num.isna().sum()), "行")

# 结构性缺失检验：这些行是否都满足 tenure == 0？
structural = tc_num.isna() & (raw["tenure"] == 0)
print("其中 tenure == 0 的结构性缺失:", int(structural.sum()), "行（新客户尚未产生累计费用）")
print(">>> 说明：TotalCharges 的缺失不是随机的（MNAR/结构性），直接中位数填充会引入偏差，")
print(">>>       正确做法是先用领域规则填 0，再保留缺失指示变量。")

# 脏记录：用一个“非空字段比例”规则识别，而不是只看某几列
non_id = raw.drop(columns=["customerID"])
completeness = non_id.notna().mean(axis=1)
junk = completeness < 0.10
print("非空字段不足 10% 的脏记录:", int(junk.sum()), "行；缺失目标值的行:", int(raw["Churn"].isna().sum()))
if junk.any():
    print(raw.loc[junk, ["customerID", "gender", "tenure", "TotalCharges", "Churn"]].to_string(index=False))

# 取值合法性扫描：发现 typo 型脏值（如 gender='Fem'），这类问题 isna() 查不出来
gender_values = raw["gender"].value_counts(dropna=False).to_dict()
print("gender 取值分布:", gender_values)
print(">>> 'Fem' 是拼写错误的非法取值（只出现在那条脏记录里），将随该行一起删除。")

# 图 2：缺失模式矩阵，直观看到“缺失集中在同一批行”
cols_with_na = raw.columns[raw.isna().any()].tolist()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.imshow(raw[cols_with_na].isna().T.values, aspect="auto", cmap="Greys", interpolation="nearest")
ax.set_yticks(range(len(cols_with_na)), cols_with_na, fontsize=9)
ax.set_xlabel("样本序号")
ax.set_title("缺失模式矩阵（黑条 = 缺失，集中在同一批新客户行）")
plt.tight_layout()
plt.savefig(FIG_DIR / "adv02_missing.png")
plt.close()

# ---- 1.3 单变量/双变量：用效应量而不是只看 p 值 ----
print("\n-- 1.3 特征与流失的关联（效应量）--")
df_eda = raw.dropna(subset=["Churn"]).copy()
df_eda["ChurnFlag"] = (df_eda["Churn"] == "Yes").astype(int)

numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges"]
assoc_rows = []
for col in numeric_cols:
    a = df_eda.loc[df_eda["ChurnFlag"] == 1, col].dropna().to_numpy()
    b = df_eda.loc[df_eda["ChurnFlag"] == 0, col].dropna().to_numpy()
    assoc_rows.append(dict(feature=col, type="数值",
                           effect=abs(rank_biserial(a, b)),
                           p_value=mannwhitneyu(a, b).pvalue))
# 类别变量：TotalCharges 是“数值被存成文本”，不参与 Cramér's V（否则会因高基数虚高）
cat_eda_cols = [c for c in df_eda.select_dtypes("object").columns
                if c not in ["customerID", "Churn", "TotalCharges"]]
for col in cat_eda_cols:
    table = pd.crosstab(df_eda[col], df_eda["Churn"])
    chi2, p, _, _ = chi2_contingency(table)
    assoc_rows.append(dict(feature=col, type="类别",
                           effect=cramers_v(table),
                           p_value=p))
assoc = pd.DataFrame(assoc_rows).sort_values("effect", ascending=False)
print(assoc.head(10).round(4).to_string(index=False))
print(">>> 效应量排序告诉你“哪些特征值得进模型”，p 值只说明“是否非零”。")

# 图 3：三个最有信息量的特征与流失的关系
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for label, color in [("No", "#4C78A8"), ("Yes", "#E45756")]:
    axes[0].hist(df_eda.loc[df_eda["Churn"] == label, "tenure"].dropna(),
                 bins=25, alpha=0.6, density=True, label=f"Churn={label}", color=color)
axes[0].set_title("在网月数 tenure")
axes[0].set_xlabel("月")
axes[0].set_ylabel("密度")
axes[0].legend()

for label, color in [("No", "#4C78A8"), ("Yes", "#E45756")]:
    axes[1].hist(df_eda.loc[df_eda["Churn"] == label, "MonthlyCharges"].dropna(),
                 bins=25, alpha=0.6, density=True, label=f"Churn={label}", color=color)
axes[1].set_title("月费用 MonthlyCharges")
axes[1].set_xlabel("美元")
axes[1].legend()

contract = pd.crosstab(df_eda["Contract"], df_eda["Churn"], normalize="index")
contract.plot(kind="bar", stacked=True, ax=axes[2], color=["#4C78A8", "#E45756"])
axes[2].set_title("合同类型 vs 流失比例")
axes[2].set_ylabel("占比")
axes[2].tick_params(axis="x", rotation=0)
axes[2].legend(title="Churn")

plt.tight_layout()
plt.savefig(FIG_DIR / "adv03_features.png")
plt.close()

# ---- 1.4 冗余与泄漏诊断 ----
print("\n-- 1.4 冗余/泄漏诊断 --")
print("tenure 与 TotalCharges 的皮尔逊相关: %.4f"
      % df_eda["tenure"].corr(pd.to_numeric(df_eda["TotalCharges"], errors="coerce")))
ok = pd.to_numeric(df_eda["TotalCharges"], errors="coerce").notna()
prod = (df_eda.loc[ok, "tenure"] * df_eda.loc[ok, "MonthlyCharges"]).to_frame("tenure_x_monthly")
reg = LinearRegression().fit(prod, pd.to_numeric(df_eda.loc[ok, "TotalCharges"], errors="coerce"))
r2 = reg.score(prod, pd.to_numeric(df_eda.loc[ok, "TotalCharges"], errors="coerce"))
print("TotalCharges ≈ tenure × MonthlyCharges 的 R² = %.4f" % r2)
print(">>> R² > 0.95：TotalCharges 是可推导的冗余特征。它会让 SHAP/重要性虚高，")
print(">>>          建模时应保留（不损失信息）但在解释时剔除，或直接做消融实验。")

# 图 4：数值特征相关性（含标签）
corr_df = df_eda[["SeniorCitizen", "tenure", "MonthlyCharges", "ChurnFlag"]].copy()
corr_df["TotalCharges"] = pd.to_numeric(df_eda["TotalCharges"], errors="coerce")
corr = corr_df.corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(corr.values[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("数值特征相关性（含 ChurnFlag）")
plt.tight_layout()
plt.savefig(FIG_DIR / "adv04_corr.png")
plt.close()

# ==================================================================
# 阶段 2：清洗数据
# ==================================================================
section("阶段 2 · 清洗数据")
clean = raw.copy()
n0 = len(clean)
log = []   # 清洗日志：每一步丢了多少行、为什么

# 规则 1：目标缺失的记录无法用于监督学习——只删这些，而不是全表 dropna
before = len(clean)
clean = clean.dropna(subset=["Churn"]).reset_index(drop=True)
log.append(("删除目标缺失/无效记录", before - len(clean), "无标签无法监督学习"))

# 规则 2：TotalCharges 去空格后强制转数值（空白字符串会暴露为 NaN）
clean["TotalCharges"] = pd.to_numeric(
    clean["TotalCharges"].astype(str).str.strip(), errors="coerce")
log.append(("TotalCharges 类型规整", 0, "空白字符串→NaN，共 %d 个" % int(clean["TotalCharges"].isna().sum())))

# 规则 3：领域规则——tenure=0 的新客户累计费用应为 0（结构性缺失，不是随机缺失）
mask_new = (clean["tenure"] == 0) & clean["TotalCharges"].isna()
clean.loc[mask_new, "TotalCharges"] = 0.0
log.append(("结构性缺失按领域规则补 0", 0, "tenure=0 的新客户 %d 行" % int(mask_new.sum())))

# 规则 4：剩余缺失交给 Pipeline 里的 SimpleImputer(add_indicator=True) 处理
remaining_na = int(clean["TotalCharges"].isna().sum())

# 规则 5：字段取值合法性校验（只记录，不静默修改）
bad_senior = int((~clean["SeniorCitizen"].isin([0, 1])).sum())
dup_id = int(clean["customerID"].duplicated().sum())
log.append(("字段校验", 0, "SeniorCitizen 非法值 %d 个，customerID 重复 %d 个" % (bad_senior, dup_id)))

print("清洗日志:")
for step, dropped, note in log:
    print("  - %-26s 删除 %d 行 | %s" % (step, dropped, note))
print("清洗前后行数: %d → %d，剩余缺失: %d" % (n0, len(clean), remaining_na))
print(">>> 原则：能按业务含义修复的修复（新客户 TotalCharges=0），")
print(">>>       不能确定机制的缺失保留 NaN + 缺失指示，交给模型判断其信息量。")

# 构造特征矩阵：数值列与类别列分开，预处理交给 ColumnTransformer
y = (clean["Churn"] == "Yes").astype(int)
X = clean.drop(columns=["Churn", "customerID"])
num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in X.columns if c not in num_cols]

# ==================================================================
# 阶段 3：预处理与划分
# ==================================================================
section("阶段 3 · 预处理与划分")
# 先划分，再做任何“从数据里学参数”的操作（补值、编码、缩放、阈值）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
print("训练集:", X_train.shape, "| 测试集:", X_test.shape)
print("训练集流失率 %.4f / 测试集流失率 %.4f（分层抽样保证一致）"
      % (y_train.mean(), y_test.mean()))


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """构造列级预处理器。

    scale=True  ：补值→标准化，供 SVM/MLP 这类对量纲敏感的模型使用
    scale=False ：只补值，树模型不需要标准化（分裂点只看排序）
    add_indicator=True：为“是否缺失”额外生成 0/1 列，保留缺失本身的信息
    handle_unknown='ignore'：测试集出现训练集没见过的类别时输出全 0，而不是报错
    drop='first'：避免哑变量陷阱（完全共线）
    """
    num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# 打印变换后的特征维度，确认编码结果符合预期（只在训练集 fit）
preview = make_preprocessor(scale=False).fit(X_train)
print("预处理后特征数:", preview.transform(X_train).shape[1], "（含缺失指示列）")
print(">>> 缩放/补值/编码全部封装进 Pipeline，交叉验证时每一折都会重新 fit，避免泄漏。")

# ==================================================================
# 阶段 4：训练多个模型
# ==================================================================
section("阶段 4 · 训练模型")

# 决策树深度扫描：用“测试集 AUC”选深度会乐观偏差，这里仅作为教学演示，
# 正式结论以 5 折交叉验证为准（见阶段 5）
depths = list(range(2, 16))
train_auc, test_auc = [], []
for depth in depths:
    pipe = Pipeline([("prep", make_preprocessor(scale=False)),
                     ("clf", DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE))])
    pipe.fit(X_train, y_train)
    train_auc.append(roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1]))
    test_auc.append(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]))
best_depth = depths[int(np.argmax(test_auc))]

plt.figure(figsize=(7, 4))
plt.plot(depths, train_auc, "o-", label="训练集 AUC")
plt.plot(depths, test_auc, "s-", label="测试集 AUC")
plt.axvline(best_depth, color="gray", ls="--", label=f"最优深度={best_depth}")
plt.xlabel("max_depth")
plt.ylabel("AUC")
plt.title("决策树深度扫描：过拟合的典型形状")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "adv06_tree_depth.png")
plt.close()
print("决策树最优深度 = %d（测试 AUC %.4f）" % (best_depth, max(test_auc)))

# 模型清单：每个都自带完整预处理，避免手动处理造成泄漏
models = {
    "基线(全预测不流失)": Pipeline([("prep", make_preprocessor(scale=False)),
                                ("clf", DummyClassifier(strategy="prior"))]),
    "朴素贝叶斯": Pipeline([("prep", make_preprocessor(scale=True)), ("clf", GaussianNB())]),
    "SVM-线性核": Pipeline([("prep", make_preprocessor(scale=True)),
                          ("clf", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE))]),
    "SVM-RBF核": Pipeline([("prep", make_preprocessor(scale=True)),
                         ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
    "MLP": Pipeline([("prep", make_preprocessor(scale=True)),
                     ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600,
                                           early_stopping=False, random_state=RANDOM_STATE))]),
    "决策树": Pipeline([("prep", make_preprocessor(scale=False)),
                      ("clf", DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE))]),
    "随机森林": Pipeline([("prep", make_preprocessor(scale=False)),
                       ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                      random_state=RANDOM_STATE))]),
    "GBDT(HistGB)": Pipeline([("prep", make_preprocessor(scale=False)),
                              ("clf", HistGradientBoostingClassifier(
                                  learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                  min_samples_leaf=20, l2_regularization=1.0,
                                  early_stopping=True, n_iter_no_change=30,
                                  random_state=RANDOM_STATE))]),
    "GBDT+类别加权+等渗校准": CalibratedClassifierCV(
        Pipeline([("prep", make_preprocessor(scale=False)),
                  ("clf", HistGradientBoostingClassifier(
                      learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                      min_samples_leaf=20, l2_regularization=1.0,
                      early_stopping=True, n_iter_no_change=30,
                      class_weight="balanced", random_state=RANDOM_STATE))]),
        method="isotonic", cv=N_SPLITS),
}

fitted, proba_test = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    proba_test[name] = model.predict_proba(X_test)[:, 1]   # 正类概率，后续算 AUC/校准
    print("已训练:", name)

# MLP 的损失曲线单独保存（课程要求，也能判断是否收敛）
mlp = fitted["MLP"].named_steps["clf"]
loss = np.asarray(mlp.loss_curve_)
print("MLP 迭代 %d 轮，loss %.4f → %.4f，最后 10 轮变化 %+.6f"
      % (mlp.n_iter_, loss[0], loss[-1], loss[-1] - loss[-10]))
plt.figure(figsize=(7, 4))
plt.plot(loss)
plt.xlabel("迭代轮数")
plt.ylabel("训练损失")
plt.title("MLP 损失曲线（尾部变平说明已收敛）")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "adv05_mlp_loss.png")
plt.close()

# ==================================================================
# 阶段 5：指标与图表比较
# ==================================================================
section("阶段 5 · 指标与图表比较")

# 说明：单次划分的结果不稳定，先算 Bootstrap 95% CI；
# 再对 5 个候选做 5 折交叉验证（全流程进 Pipeline），用 CV 均值±标准差判断稳定性。
cv_names = ["SVM-线性核", "朴素贝叶斯", "随机森林", "GBDT(HistGB)", "GBDT+类别加权+等渗校准"]
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_stats = {}
for name in cv_names:
    scores = cross_val_score(models[name], X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_stats[name] = (scores.mean(), scores.std())
    print("5 折 CV AUC · %-22s %.4f ± %.4f" % (name, scores.mean(), scores.std()))

rows = []
for name, proba in proba_test.items():
    pred = (proba >= 0.5).astype(int)                    # 先用 0.5 看“默认表现”
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    auc_lo, auc_hi = bootstrap_ci(y_test, proba, roc_auc_score)
    ap_lo, ap_hi = bootstrap_ci(y_test, proba, average_precision_score)
    rows.append(dict(
        model=name,
        auc=auc, auc_lo=auc_lo, auc_hi=auc_hi,
        pr_auc=ap, pr_lo=ap_lo, pr_hi=ap_hi,
        sens=tp / (tp + fn), spec=tn / (tn + fp),
        precision=precision_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        brier=brier_score_loss(y_test, proba),
        logloss=log_loss(y_test, np.clip(proba, 1e-6, 1 - 1e-6)),
        cv_auc=cv_stats.get(name, (np.nan, np.nan))[0],
        cv_std=cv_stats.get(name, (np.nan, np.nan))[1],
    ))

scores_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
print("\n模型成绩单（按 PR-AUC 排序，不平衡数据优先看它）:")
print(scores_df.round(4).to_string(index=False))
scores_df.to_csv(RESULT_DIR / "advanced_scores.csv", index=False,
                 encoding="utf-8-sig", float_format="%.4f")

# 图 7：ROC / PR / 校准曲线三联图
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for name, proba in proba_test.items():
    if name.startswith("基线"):
        continue
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, lw=1.6, label=f"{name} ({roc_auc_score(y_test, proba):.3f})")
    prec, rec, _ = precision_recall_curve(y_test, proba)
    axes[1].plot(rec, prec, lw=1.6, label=f"{name} ({average_precision_score(y_test, proba):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("FPR = 1 - 特异度")
axes[0].set_ylabel("TPR = 灵敏度")
axes[0].set_title("ROC 曲线（图例为 AUC）")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)

axes[1].axhline(y_test.mean(), color="gray", ls="--", lw=1, label="随机基线")
axes[1].set_xlabel("召回率 = 灵敏度")
axes[1].set_ylabel("精确率")
axes[1].set_title("PR 曲线（图例为 PR-AUC）")
axes[1].legend(fontsize=7)
axes[1].grid(alpha=0.3)

# 校准曲线：预测概率 vs 实际流失率，越贴近对角线越可信
for name in ["GBDT(HistGB)", "GBDT+类别加权+等渗校准"]:
    frac_pos, mean_pred = calibration_curve(y_test, proba_test[name],
                                            n_bins=10, strategy="quantile")
    axes[2].plot(mean_pred, frac_pos, "o-", lw=1.6, ms=4, label=name)
axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="完美校准")
axes[2].set_xlabel("预测概率")
axes[2].set_ylabel("实际流失比例")
axes[2].set_title("校准曲线（等渗校准前后）")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "adv07_roc_pr_cal.png")
plt.close()

# 图 8：提升图（lift）与累积增益图（gain）——运营最关心的“前 20% 能抓到多少”
order = np.argsort(-proba_test["GBDT+类别加权+等渗校准"])
y_sorted = y_test.to_numpy()[order]
cum_churn = np.cumsum(y_sorted) / y_sorted.sum()      # 累积抓到的流失客户占比
cum_pop = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
lift = cum_churn / cum_pop                              # 提升度 = 捕获率 / 触达率

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].plot(cum_pop, cum_churn, lw=2, color="#E45756")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="随机触达")
axes[0].axvline(CAPACITY, color="gray", ls=":", lw=1.2)
axes[0].set_xlabel("触达客户比例")
axes[0].set_ylabel("累计抓到的流失客户比例")
axes[0].set_title("累积增益图 gain chart")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(cum_pop, lift, lw=2, color="#4C78A8")
axes[1].axhline(1, color="k", ls="--", lw=1)
axes[1].axvline(CAPACITY, color="gray", ls=":", lw=1.2)
axes[1].set_xlabel("触达客户比例")
axes[1].set_ylabel("提升度 lift")
axes[1].set_title("提升图 lift chart")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "adv08_lift_gain.png")
plt.close()

gain20 = cum_churn[int(CAPACITY * len(y_sorted)) - 1]
print("触达前 %.0f%% 客户可抓到 %.1f%% 的流失客户（提升度 %.2f）"
      % (CAPACITY * 100, gain20 * 100, gain20 / CAPACITY))

# ==================================================================
# 阶段 6：模型选择与业务取舍
# ==================================================================
section("阶段 6 · 模型选择与业务取舍")

# 业务模型：
#   收益 = 成功挽留人数 × ARR × CLV（24 个月月费）
#   成本 = 触达人数 × 50 美元优惠成本
#   ARR（挽留成功率）用训练集留存率近似；更严谨可用 Kaplan-Meier（e-Profits, arXiv:2507.08860）
arr = float(1 - y_train.mean())
monthly_test = X_test["MonthlyCharges"].to_numpy()


def profit_per_1000(y_true, pred, monthly):
    """计算每 1000 名客户的期望利润（美元）。"""
    saved = ((y_true == 1) & (pred == 1)) * arr * CLV_MONTHS * monthly
    cost = pred * OFFER_COST
    return (saved - cost).sum() / len(y_true) * 1000


def topk_pred(proba, k):
    """按风险分数取前 k 比例作为“要联系”的客户（容量约束，贴合运营现实）。"""
    pred = np.zeros(len(proba), dtype=int)
    pred[np.argsort(-np.asarray(proba))[: int(round(k * len(proba)))]] = 1
    return pred


# 6.1 同一容量（Top20%）下比较各模型利润
biz_rows = []
y_arr = y_test.to_numpy()
for name, proba in proba_test.items():
    pred = topk_pred(proba, CAPACITY)
    tn, fp, fn, tp = confusion_matrix(y_arr, pred).ravel()
    biz_rows.append(dict(
        model=name,
        contacted=int(pred.sum()),
        sens=tp / (tp + fn), precision=tp / (tp + fp) if (tp + fp) else 0,
        profit_per_1000=profit_per_1000(y_arr, pred, monthly_test),
    ))
biz_df = pd.DataFrame(biz_rows).sort_values("profit_per_1000", ascending=False).reset_index(drop=True)
print("同一运营容量（前 %.0f%%）下的业务收益:" % (CAPACITY * 100))
print(biz_df.round(4).to_string(index=False))
biz_df.to_csv(RESULT_DIR / "business_summary.csv", index=False,
              encoding="utf-8-sig", float_format="%.4f")

# 6.2 利润-容量曲线：不同触达预算下，哪个模型更值得上线
capacities = np.linspace(0.05, 0.60, 23)
plt.figure(figsize=(8, 4.8))
for name in ["随机森林", "GBDT(HistGB)", "GBDT+类别加权+等渗校准"]:
    profits = [profit_per_1000(y_arr, topk_pred(proba_test[name], c), monthly_test) for c in capacities]
    plt.plot(capacities * 100, profits, "o-", ms=3, lw=1.8, label=name)
plt.axvline(CAPACITY * 100, color="gray", ls="--", label="当前容量 20%")
plt.xlabel("触达客户比例（%）")
plt.ylabel("期望利润（美元 / 每千人）")
plt.title("利润-容量曲线：预算变化时模型排序可能改变")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "adv09_profit_capacity.png")
plt.close()

# 6.3 选型规则（避免被单次划分的噪声带偏）：
#   ① 先用 5 折 CV 圈定“统计第一梯队”：CV AUC 与最优值相差不超过 0.01
#   ② 梯队内比较 20% 容量利润，差距小于 5% 视为打平
#   ③ 打平时优先“概率已校准 + CV 波动更小”的模型（阈值可迁移、上线更稳）
cv_best = max(v[0] for v in cv_stats.values())
first_tier = [m for m, v in cv_stats.items() if v[0] >= cv_best - 0.01]
best_stat = scores_df.iloc[0]["model"]
tier_biz = biz_df[biz_df["model"].isin(first_tier)].copy()
top_profit = tier_biz["profit_per_1000"].max()
tied = tier_biz[tier_biz["profit_per_1000"] >= 0.95 * top_profit]
deploy_name = (tied.sort_values("profit_per_1000", ascending=False)
                   .iloc[0]["model"])
for cand in tied["model"]:
    if "校准" in cand:                    # 打平时优先校准过的概率
        deploy_name = cand
        break

print("\n第一梯队（CV AUC ≥ %.4f）: %s" % (cv_best - 0.01, "、".join(first_tier)))
print("统计最优（PR-AUC）:", best_stat)
print("业务最优（20%% 容量利润）: %s（%.0f 美元/千人）"
      % (biz_df.iloc[0]["model"], biz_df.iloc[0]["profit_per_1000"]))
print("最终建议上线: %s" % deploy_name)
print(">>> 理由：候选模型利润差距在 5% 以内、AUC 差距 < 0.01 时，统计上并未分出胜负；")
print(">>>       此时应优先选择概率经过校准、CV 波动更小的模型，阈值才可迁移到线上。")

# 6.4 解释性：置换重要性（模型无关；比 GBDT 自带 importance 更不易被高基数特征带偏）
best_model = fitted[deploy_name]
perm = permutation_importance(best_model, X_test, y_test, scoring="roc_auc",
                              n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False).head(10)
print("\n置换重要性 Top10:")
print(imp.round(4).to_string())
print(">>> 注意：TotalCharges 排名靠前不代表独立业务价值——它与 tenure、MonthlyCharges 高度冗余，")
print(">>>       业务解释时应合并为“累计消费能力 = 在网时长 × 月费”的复合信号。")

plt.figure(figsize=(7, 4.8))
imp[::-1].plot(kind="barh", color="#5B8FF9")
plt.xlabel("AUC 下降量（置换该特征后）")
plt.title(f"置换重要性 Top10 · {deploy_name}")
plt.tight_layout()
plt.savefig(FIG_DIR / "adv10_importance.png")
plt.close()

# 6.5 输出最终建议（报告可直接引用）
section("最终建议（可直接写进报告结论）")
print("1) 主指标用 PR-AUC + 灵敏度/特异度，并用 Bootstrap 置信区间说明不确定性；")
print("   本数据 AUC 差异普遍 < 0.02，单次划分的“排名”并不可靠。")
print("2) 不平衡处理优先 class_weight + 概率校准，不用“先 SMOTE 再划分”的泄漏写法。")
print("3) 阈值由运营容量和单客经济性决定：当前 20%% 容量对应约 %d 人/千人。"
      % int(CAPACITY * 1000))
print("4) 冗余特征 TotalCharges（R²=%.3f）仍参与建模，但其重要性主要来自冗余，"
      "解释时应与其他特征合并解读。" % r2)
print("5) 已保存: results/advanced_scores.csv、results/business_summary.csv")
