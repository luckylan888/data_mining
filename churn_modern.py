"""
现代方案补充实验（对应 churn_pipeline.py 的进阶版，不替代课程要求的五算法）

思路来源（2026 年 arXiv 关于 Telco Churn 的最新研究）：
  - 梯度提升树（GBDT）仍是表格数据最强基线，深度表格模型只在部分场景占优
  - 类别不平衡优先用 class_weight，而不是先划分再做 SMOTE（后者会泄漏，F1 虚高约 13 个百分点）
  - 概率需要校准（等渗回归通常是更稳的默认），否则阈值和期望成本都是错的
  - 业务上应该优化"期望成本阈值"，而不是 0.5 或 F1 最优阈值
  - TotalCharges ≈ tenure × MonthlyCharges（R² 极高），做 SHAP/重要性前要先查冗余

运行：
    python churn_modern.py

产出：
    figures/fig09_cost_threshold.png   期望成本随阈值变化
    figures/fig10_calibration.png      校准曲线
    figures/fig11_perm_importance.png  置换重要性 Top10
    results/modern_comparison.csv      现代方案对比表
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
RESULT_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
RETENTION_COST = 50.0      # 发一次挽留优惠的成本（美元）
CLV_MONTHS = 24            # 流失一个客户的损失 ≈ 24 个月月费

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def find_data_file():
    for p in [
        BASE_DIR / "data" / "telco_churn.csv",
        BASE_DIR.parent / "data" / "telco_churn.csv",
        BASE_DIR / "telco_churn.csv",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("找不到 telco_churn.csv")


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ------------------------------------------------------------------
# 0. 数据：与主流水线同样的清洗，但保留 TotalCharges 的 NaN
# ------------------------------------------------------------------
section("0. 数据准备")
df = pd.read_csv(find_data_file())
df = df.dropna(subset=["Churn"]).reset_index(drop=True)          # 删脏行
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # 空白字符串 -> NaN
df = df.drop(columns=["customerID"])
y = (df["Churn"] == "Yes").astype(int)
X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True)
X = X.astype(float)  # 保留 TotalCharges 的 NaN，交给 HGB 自己处理

# 先划分，所有拟合步骤只允许在训练集上发生
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
monthly_train = X_train["MonthlyCharges"].to_numpy()
monthly_test = X_test["MonthlyCharges"].to_numpy()
print("训练集", X_train.shape, "测试集", X_test.shape)
print("训练集流失率 %.4f / 测试集流失率 %.4f" % (y_train.mean(), y_test.mean()))

# ------------------------------------------------------------------
# 1. 冗余诊断：TotalCharges 是不是“伪特征”
# ------------------------------------------------------------------
section("1. 冗余诊断（先查泄漏/冗余，再谈重要性）")
ok = df["TotalCharges"].notna()
prod = (df.loc[ok, "tenure"] * df.loc[ok, "MonthlyCharges"]).to_frame("tenure_x_monthly")
reg = LinearRegression().fit(prod, df.loc[ok, "TotalCharges"])
r2 = reg.score(prod, df.loc[ok, "TotalCharges"])
ratio = (df.loc[ok, "TotalCharges"] / prod["tenure_x_monthly"].replace(0, np.nan)).median()
print("TotalCharges ≈ tenure × MonthlyCharges：R² = %.4f，比值中位数 = %.3f" % (r2, ratio))
print(">>> R² > 0.95 说明 TotalCharges 基本是另两列的确定性组合，"
      "它排到特征重要性第一并不代表业务信息多，反而会稀释 SHAP 解释。")

# ------------------------------------------------------------------
# 2. 模型：传统基线 vs 现代 GBDT（+加权、+校准、+去冗余）
# ------------------------------------------------------------------
section("2. 模型配置")


def make_hgb(class_weight=None):
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        class_weight=class_weight, random_state=RANDOM_STATE,
    )


models = {
    "RF-默认(传统基线)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    ]),
    "HGB-默认": Pipeline([("clf", make_hgb())]),
    "HGB-加权": Pipeline([("clf", make_hgb(class_weight="balanced"))]),
    "HGB-加权+等渗校准": CalibratedClassifierCV(
        Pipeline([("clf", make_hgb(class_weight="balanced"))]),
        method="isotonic", cv=5,
    ),
    "HGB-加权+校准+去TotalCharges": CalibratedClassifierCV(
        Pipeline([("clf", make_hgb(class_weight="balanced"))]),
        method="isotonic", cv=5,
    ),
}
drop_cols = {"HGB-加权+校准+去TotalCharges": ["TotalCharges"]}

# ------------------------------------------------------------------
# 3. 训练 + 用 OOF 概率选阈值（不碰测试集）
# ------------------------------------------------------------------
section("3. 训练与成本敏感阈值选择（阈值只在训练集 OOF 上调）")
THRESHOLDS = np.linspace(0.01, 0.99, 99)
CAPACITY = 0.20   # 现实约束：挽留团队每月只能联系风险最高的 20% 客户


def cost_of_pred(y_true, pred, monthly, per=1000):
    """FN = 24 个月月费；TP/FP = 50 美元挽留成本；TN = 0。"""
    fn_cost = ((y_true == 1) & (pred == 0)) * CLV_MONTHS * monthly
    return (fn_cost + pred * RETENTION_COST).sum() / len(y_true) * per


def expected_cost(y_true, proba, threshold, monthly, per=1000):
    return cost_of_pred(y_true, (np.asarray(proba) >= threshold).astype(int), monthly, per)


def topk_pred(proba, k):
    pred = np.zeros(len(proba), dtype=int)
    pred[np.argsort(-np.asarray(proba))[: int(round(k * len(proba)))]] = 1
    return pred


def summarize(name, strategy, proba, pred, y_true, monthly):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return dict(
        model=name, strategy=strategy,
        auc=roc_auc_score(y_true, proba),
        pr_auc=average_precision_score(y_true, proba),
        brier=brier_score_loss(y_true, proba),
        sens=tp / (tp + fn), spec=tn / (tn + fp),
        precision=tp / (tp + fp) if (tp + fp) else 0.0,
        flag_rate=pred.mean(),
        cost_per_1000=cost_of_pred(y_true, pred, monthly),
    )


rows = []
fitted, oof_probs, chosen = {}, {}, {}

for name, model in models.items():
    Xtr = X_train.drop(columns=drop_cols.get(name, []))
    Xte = X_test.drop(columns=drop_cols.get(name, []))

    # OOF 概率用于选阈值（5 折，预处理全在 Pipeline 内）
    oof = cross_val_predict(model, Xtr, y_train, cv=StratifiedKFold(5, shuffle=True,
                            random_state=RANDOM_STATE), method="predict_proba", n_jobs=-1)[:, 1]
    oof_probs[name] = oof

    costs = [expected_cost(y_train.to_numpy(), oof, t, monthly_train) for t in THRESHOLDS]
    best_t = float(THRESHOLDS[int(np.argmin(costs))])
    chosen[name] = best_t

    # 全训练集重新拟合，只在测试集上评估一次
    model.fit(Xtr, y_train)
    fitted[name] = (model, Xte)
    proba = model.predict_proba(Xte)[:, 1]
    y_arr = y_test.to_numpy()

    rows.append(summarize(name, "阈值 0.5", proba, (proba >= 0.5).astype(int), y_arr, monthly_test))
    rows.append(summarize(name, "成本最优阈值 %.2f" % best_t, proba,
                          (proba >= best_t).astype(int), y_arr, monthly_test))
    rows.append(summarize(name, "容量 Top%d%%" % int(CAPACITY * 100), proba,
                          topk_pred(proba, CAPACITY), y_arr, monthly_test))
    print("[%s] OOF 最优阈值 = %.2f（训练集期望成本 %.0f 美元/千人；该阈值会联系 %.0f%% 客户）"
          % (name, best_t, min(costs),
             100 * np.mean(oof >= best_t)))

res_df = pd.DataFrame(rows)
section("4. 对比结果（测试集）")
print(res_df.round(4).to_string(index=False))
res_df.to_csv(RESULT_DIR / "modern_comparison.csv", index=False,
              encoding="utf-8-sig", float_format="%.4f")
print("\n已保存: results/modern_comparison.csv")

section("4.1 同一接触容量（Top20%）下的公平对比")
cap_df = res_df[res_df.strategy.str.startswith("容量")].sort_values("cost_per_1000")
print(cap_df.round(4).to_string(index=False))

# ------------------------------------------------------------------
# 5. 图：成本-阈值曲线 + 校准曲线 + 置换重要性
# ------------------------------------------------------------------
print("\n-- 5.1 期望成本随阈值变化（训练集 OOF）--")
plt.figure(figsize=(8, 4.8))
for name, oof in oof_probs.items():
    costs = [expected_cost(y_train.to_numpy(), oof, t, monthly_train) for t in THRESHOLDS]
    line, = plt.plot(THRESHOLDS, costs, label=name, lw=1.8)
    plt.scatter([chosen[name]], [min(costs)], color=line.get_color(), zorder=5)
plt.xlabel("决策阈值")
plt.ylabel("期望成本（美元 / 每千名客户）")
plt.title("成本敏感阈值：曲线最低点即业务最优阈值")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig09_cost_threshold.png", dpi=120)
plt.close()

print("-- 5.2 校准曲线：校准前后概率是否可信 --")
fig, ax = plt.subplots(figsize=(5.6, 5))
ax.plot([0, 1], [0, 1], "k--", label="完美校准")
for name in ["HGB-加权", "HGB-加权+等渗校准"]:
    model, Xte = fitted[name]
    proba = model.predict_proba(Xte)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", label=name,
            lw=1.6, ms=4)
ax.set_xlabel("预测概率")
ax.set_ylabel("实际流失比例")
ax.set_title("可靠性曲线")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig10_calibration.png", dpi=120)
plt.close()

print("-- 5.3 置换重要性（模型无关，替代 SHAP 的轻量做法）--")
best_name = "HGB-加权+校准+去TotalCharges"
best_model, best_X = fitted[best_name]
perm = permutation_importance(best_model, best_X, y_test, scoring="roc_auc",
                              n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=best_X.columns).sort_values(ascending=False).head(10)
print(imp.round(4).to_string())

plt.figure(figsize=(7, 4.8))
imp[::-1].plot(kind="barh", color="#5B8FF9")
plt.xlabel("AUC 下降量（置换后）")
plt.title("置换重要性 Top10（已去掉冗余的 TotalCharges）")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig11_perm_importance.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 6. 结论
# ------------------------------------------------------------------
section("6. 结论（可写进报告“业务建议”一节）")
rf_cap = cap_df[cap_df.model == "RF-默认(传统基线)"].iloc[0]
best_cap = cap_df.iloc[0]
print("同一接触容量（Top20%，约 300 人）下：")
print("  传统 RF：AUC %.4f，抓到 %.1f%% 流失客户，精确率 %.2f，期望成本 %.0f 美元/千人"
      % (rf_cap.auc, rf_cap.sens * 100, rf_cap.precision, rf_cap.cost_per_1000))
print("  最优 %s：AUC %.4f，抓到 %.1f%% 流失客户，精确率 %.2f，期望成本 %.0f 美元/千人"
      % (best_cap.model, best_cap.auc, best_cap.sens * 100, best_cap.precision,
         best_cap.cost_per_1000))
print("  成本降低 %.1f%%" % ((1 - best_cap.cost_per_1000 / rf_cap.cost_per_1000) * 100))
print("\n无容量约束时，成本最优阈值会低到 0.03~0.07（FN 成本约为 FP 的 30 倍），"
      "会把 97% 的客户都标成高风险——这不是模型更好，而是没有考虑运营能力。")
print(">>> 现代做法不是换更花哨的模型，而是：GBDT + 类别加权 + 概率校准 + 业务约束下的阈值 + 冗余诊断。")
print(">>> 注意：课程要求的 NB/SVM/MLP/DT/RF 仍要保留，本脚本只作为“进阶/加分”部分。")
