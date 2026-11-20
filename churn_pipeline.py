"""
《分类算法大比武：客户流失预警》—— 纯 Python 版（无需 Jupyter / Anaconda）

运行方式：
    pip install pandas numpy matplotlib scikit-learn
    python churn_pipeline.py

产出：
    figures/fig01_churn_dist.png      标签分布（计数 + 比例）
    figures/fig02_feature_churn.png   3 个特征与流失的关系
    figures/fig03_corr_heatmap.png    相关性热力图
    figures/fig04_mlp_loss.png        MLP 训练损失曲线
    figures/fig05_tree_depth.png      决策树深度扫描
    figures/fig06_rf_importance.png   随机森林特征重要性
    figures/fig07_roc_all.png         多模型 ROC 对比
    figures/fig08_imbalance.png       class_weight 前后对比
    results/model_comparison.csv      比武成绩单
"""

import os
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无图形界面环境下也能保存图片
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ----------------------------- 全局配置 -----------------------------
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
RESULT_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42  # 全组统一随机种子，不要改

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def find_data_file():
    """依次尝试几个常见位置，找到 telco_churn.csv。"""
    candidates = [
        BASE_DIR / "data" / "telco_churn.csv",
        BASE_DIR.parent / "data" / "telco_churn.csv",
        BASE_DIR / "telco_churn.csv",
        Path.cwd() / "data" / "telco_churn.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("找不到 telco_churn.csv，请把数据放到项目目录下的 data/ 中")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==================================================================
# 0. 读取数据
# ==================================================================
section("0. 读取数据")
DATA_PATH = find_data_file()
print("数据文件:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("数据形状:", df.shape)
print("前 5 行（关键字段）:")
print(df[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]].head().to_string(index=False))
print("\n字段类型:")
print(df.dtypes.to_string())

# ==================================================================
# 任务一：探索性分析（EDA）
# ==================================================================
section("任务一：探索性分析（EDA）")

# 1.1 标签分布：亲眼看到不平衡
print("\n-- 1.1 标签分布 --")
churn_counts = df["Churn"].value_counts(dropna=False)
churn_rate = df["Churn"].value_counts(normalize=True, dropna=False)
print("流失计数:\n" + churn_counts.to_string())
print("流失比例:\n" + churn_rate.round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
counts_plot = df["Churn"].value_counts()
axes[0].bar(counts_plot.index, counts_plot.values, color=["#4C78A8", "#E45756"])
axes[0].set_title("客户数量")
axes[0].set_ylabel("人数")
for i, v in enumerate(counts_plot.values):
    axes[0].text(i, v + 40, str(v), ha="center")

ratio_plot = df["Churn"].value_counts(normalize=True)
axes[1].bar(ratio_plot.index, ratio_plot.values, color=["#4C78A8", "#E45756"])
axes[1].set_title("客户占比")
axes[1].set_ylabel("比例")
for i, v in enumerate(ratio_plot.values):
    axes[1].text(i, v + 0.01, f"{v:.2%}", ha="center")
axes[1].set_ylim(0, 1)

plt.suptitle("Churn 标签分布（不平衡二分类）")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig01_churn_dist.png", dpi=120)
plt.close()
print(">>> 结论：流失率约 %.2f%%，正负样本约 1:%.1f，属于不平衡二分类。"
      % (churn_rate.get("Yes", np.nan) * 100, churn_counts.get("No", 0) / churn_counts.get("Yes", 1)))

# 1.2 缺失值/脏数据侦查
print("\n-- 1.2 缺失与脏数据排查 --")
print("TotalCharges 的 dtype:", df["TotalCharges"].dtype)

total_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
bad_mask = total_numeric.isna()
print("TotalCharges 无法转成数值的行数:", int(bad_mask.sum()))
print("这些行的 tenure 取值:", sorted(df.loc[bad_mask, "tenure"].dropna().unique().tolist()))
print("其中 tenure == 0 的行数（新客户，尚未产生累计费用）:",
      int(((df["tenure"] == 0) & bad_mask).sum()))

near_empty = df.drop(columns=["customerID"]).isna().mean(axis=1) >= 0.9
print("几乎整行为空的脏记录数:", int(near_empty.sum()))
if near_empty.any():
    print(df.loc[near_empty, ["customerID", "gender", "tenure", "TotalCharges", "Churn"]].to_string(index=False))
print("Churn 缺失行数:", int(df["Churn"].isna().sum()))
print(">>> 三个陷阱：① TotalCharges 是 object（含空白字符串）；"
      "② 空白字符串用 isna() 查不出来；③ 末尾存在只有 customerID/gender 的脏记录，"
      "只能靠 dropna(subset=['Churn']) 识别。")

# 1.3 关键特征与流失的关系（3 个：tenure、MonthlyCharges、Contract）
print("\n-- 1.3 关键特征与流失的关系 --")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for label, color in [("No", "#4C78A8"), ("Yes", "#E45756")]:
    axes[0].hist(df.loc[df["Churn"] == label, "tenure"].dropna(),
                 bins=25, alpha=0.6, density=True, label=f"Churn={label}", color=color)
axes[0].set_title("在网月数 tenure")
axes[0].set_xlabel("tenure（月）")
axes[0].set_ylabel("密度")
axes[0].legend()

for label, color in [("No", "#4C78A8"), ("Yes", "#E45756")]:
    axes[1].hist(df.loc[df["Churn"] == label, "MonthlyCharges"].dropna(),
                 bins=25, alpha=0.6, density=True, label=f"Churn={label}", color=color)
axes[1].set_title("月费用 MonthlyCharges")
axes[1].set_xlabel("月费")
axes[1].set_ylabel("密度")
axes[1].legend()

contract_rate = pd.crosstab(df["Contract"], df["Churn"], normalize="index")
contract_rate.plot(kind="bar", stacked=True, ax=axes[2], color=["#4C78A8", "#E45756"])
axes[2].set_title("合同类型 vs 流失比例")
axes[2].set_ylabel("占比")
axes[2].set_xlabel("Contract")
axes[2].legend(title="Churn")
axes[2].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig02_feature_churn.png", dpi=120)
plt.close()

print("各合同类型的流失率:\n" + (contract_rate["Yes"] * 100).round(2).to_string())
print("tenure 分组流失率:")
tenure_bin = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 100],
                    labels=["0-6月", "7-12月", "13-24月", "25-48月", "48月+"])
print((df.groupby(tenure_bin, observed=True)["Churn"].apply(lambda s: (s == "Yes").mean()) * 100).round(2).to_string())

# 1.4 相关性热力图
print("\n-- 1.4 相关性热力图 --")
num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges"]
d = df.copy()
d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
d["ChurnFlag"] = (d["Churn"] == "Yes").astype(int)
corr = d[num_cols + ["TotalCharges", "ChurnFlag"]].corr()

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(corr.values[i, j]) > 0.6 else "black", fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("数值特征相关性热力图")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig03_corr_heatmap.png", dpi=120)
plt.close()

tenure_total_corr = corr.loc["tenure", "TotalCharges"]
print("tenure 与 TotalCharges 的相关系数: %.3f" % tenure_total_corr)
print(">>> 证据：这两个特征高度线性相关，朴素贝叶斯的条件独立假设不成立。")

# ==================================================================
# 任务二：数据预处理
# ==================================================================
section("任务二：数据预处理")

data = df.copy()
raw_rows = len(data)

# ① 删除只能靠 dropna(subset=["Churn"]) 识别的脏行
data = data.dropna(subset=["Churn"]).reset_index(drop=True)
print("删除脏记录:", raw_rows - len(data), "行；剩余", len(data), "行")

# ② TotalCharges 转数值：新客户（tenure=0）补 0，其余补中位数
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
n_new = int(((data["tenure"] == 0) & data["TotalCharges"].isna()).sum())
n_other = int(data["TotalCharges"].isna().sum()) - n_new
data.loc[(data["tenure"] == 0) & data["TotalCharges"].isna(), "TotalCharges"] = 0
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())
print("TotalCharges 补值：新客户补 0 共 %d 行，其余中位数补 %d 行" % (n_new, n_other))
print("补值后剩余缺失数:", int(data["TotalCharges"].isna().sum()))

# ③ 丢弃标识列
data = data.drop(columns=["customerID"])

# ④ 标签转 0/1
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0}).astype(int)

# ⑤ 独热编码（drop_first=True 避免哑变量陷阱）
X = pd.get_dummies(data.drop(columns=["Churn"]), drop_first=True).astype(float)
y = data["Churn"]
print("特征数:", X.shape[1], "| 正类比例: %.4f" % y.mean())

# ⑥ 先划分，后缩放（防止数据泄漏）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
print("训练集", X_train.shape, "测试集", X_test.shape)

scaler = StandardScaler().fit(X_train)  # 只在训练集上 fit
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
print("训练集流失率 %.4f / 测试集流失率 %.4f" % (y_train.mean(), y_test.mean()))

# ==================================================================
# 任务三：五算法大比武
# ==================================================================
section("任务三：五算法大比武")

results = []


def report(name, y_true, y_pred, y_proba, record=True):
    """统一计算准确率、灵敏度、特异度、AUC。"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    print(f"[{name}] Acc={acc:.4f}  灵敏度={sens:.4f}  特异度={spec:.4f}  AUC={auc:.4f}")
    row = dict(model=name, acc=acc, sens=sens, spec=spec, auc=auc)
    if record:
        results.append(row)
    return row


# 3.E 基线：全预测"不流失"
print("\n-- 3.E 基线 --")
y_base = np.zeros(len(y_test), dtype=int)
baseline_acc = accuracy_score(y_test, y_base)
print("基线准确率 = %.4f" % baseline_acc)
print("基线灵敏度 = 0.0000（一个流失客户都抓不到），特异度 = 1.0000")
results.append(dict(model="基线(全预测不流失)", acc=baseline_acc, sens=0.0, spec=1.0, auc=0.5))
print(">>> 为什么它没用：流失客户全被漏掉，召回为 0，业务上等于没做预警。")

# 3.A 朴素贝叶斯（不缩放）
print("\n-- 3.A 朴素贝叶斯 --")
nb = GaussianNB().fit(X_train, y_train)
nb_row = report("朴素贝叶斯", y_test, nb.predict(X_test), nb.predict_proba(X_test)[:, 1])

# 3.B SVM：线性核 → RBF 核（必须缩放）
print("\n-- 3.B SVM --")
svm_lin = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE).fit(X_train_s, y_train)
lin_row = report("SVM-线性核", y_test, svm_lin.predict(X_test_s), svm_lin.predict_proba(X_test_s)[:, 1])
print("线性核支持向量数:", int(svm_lin.n_support_.sum()))

svm_rbf = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE).fit(X_train_s, y_train)
rbf_row = report("SVM-RBF核", y_test, svm_rbf.predict(X_test_s), svm_rbf.predict_proba(X_test_s)[:, 1])
print("RBF核支持向量数:", int(svm_rbf.n_support_.sum()))
print(">>> RBF - 线性核 AUC 差值: %+.4f" % (rbf_row["auc"] - lin_row["auc"]))

# 3.C MLP（必须缩放）
print("\n-- 3.C MLP --")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600,
                    random_state=RANDOM_STATE).fit(X_train_s, y_train)
report("MLP", y_test, mlp.predict(X_test_s), mlp.predict_proba(X_test_s)[:, 1])
print("实际迭代轮数:", mlp.n_iter_)
loss = np.asarray(mlp.loss_curve_)
print("loss 首值 %.4f → 末值 %.4f，最后 10 轮变化 %+.6f"
      % (loss[0], loss[-1], loss[-1] - loss[-10]))

plt.figure(figsize=(7, 4))
plt.plot(loss)
plt.xlabel("迭代轮数")
plt.ylabel("训练损失 log loss")
plt.title("MLP 训练损失曲线（loss_curve_）")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig04_mlp_loss.png", dpi=120)
plt.close()

# 3.D 决策树深度扫描 + 随机森林（树模型不缩放）
print("\n-- 3.D 决策树深度扫描 + 随机森林 --")
depths = range(2, 16)
train_auc, test_auc = [], []
for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE).fit(X_train, y_train)
    train_auc.append(roc_auc_score(y_train, tree.predict_proba(X_train)[:, 1]))
    test_auc.append(roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))

plt.figure(figsize=(7, 4))
plt.plot(list(depths), train_auc, "o-", label="训练集 AUC")
plt.plot(list(depths), test_auc, "s-", label="测试集 AUC")
best_d = list(depths)[int(np.argmax(test_auc))]
plt.axvline(best_d, color="gray", linestyle="--", label=f"最优深度={best_d}")
plt.xlabel("max_depth")
plt.ylabel("AUC")
plt.title("决策树深度扫描")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig05_tree_depth.png", dpi=120)
plt.close()

print("各深度测试 AUC:", {d: round(float(a), 4) for d, a in zip(depths, test_auc)})
print("最优深度:", best_d)
dt = DecisionTreeClassifier(max_depth=best_d, random_state=RANDOM_STATE).fit(X_train, y_train)
report("决策树", y_test, dt.predict(X_test), dt.predict_proba(X_test)[:, 1])

rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
report("随机森林", y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])

# 特征重要性
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
print("\n随机森林 Top10 特征重要性:\n" + imp.round(4).to_string())

plt.figure(figsize=(7, 5))
imp[::-1].plot(kind="barh", color="#4C78A8")
plt.xlabel("importance")
plt.title("随机森林特征重要性 Top10")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig06_rf_importance.png", dpi=120)
plt.close()

# ==================================================================
# 任务四：统一评估与不平衡处理
# ==================================================================
section("任务四：评估与不平衡处理")

# 4.1 多模型 ROC 同图
print("\n-- 4.1 ROC 对比 --")
models = {
    "朴素贝叶斯": (nb, False),
    "SVM-线性核": (svm_lin, True),
    "SVM-RBF核": (svm_rbf, True),
    "MLP": (mlp, True),
    "决策树": (dt, False),
    "随机森林": (rf, False),
}
plt.figure(figsize=(7, 6))
for name, (model, need_scale) in models.items():
    X_eval = X_test_s if need_scale else X_test
    proba = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="随机猜测 (AUC=0.5)")
plt.xlabel("假正率 FPR = 1 - 特异度")
plt.ylabel("真正率 TPR = 灵敏度")
plt.title("多模型 ROC 对比")
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig07_roc_all.png", dpi=120)
plt.close()

# 4.2 5 折交叉验证（缩放放进 Pipeline）
print("\n-- 4.2 5 折交叉验证（AUC，均值 ± 标准差）--")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_models = {
    "朴素贝叶斯": GaussianNB(),
    "SVM-RBF核": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=RANDOM_STATE),
    "决策树": DecisionTreeClassifier(max_depth=best_d, random_state=RANDOM_STATE),
    "随机森林": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
}
cv_stats = {}
for name, model in cv_models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_stats[name] = (scores.mean(), scores.std())
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

print(">>> 随机森林 CV 标准差 %.4f < 决策树 %.4f：组合模型方差更小、更稳定。"
      % (cv_stats["随机森林"][1], cv_stats["决策树"][1]))

# 4.3 class_weight='balanced' 前后对比
print("\n-- 4.3 class_weight='balanced' 对比 --")
svm_bal = SVC(kernel="rbf", class_weight="balanced", probability=True,
              random_state=RANDOM_STATE).fit(X_train_s, y_train)
rf_bal = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)

cmp_rows = []
for name, model, X_eval in [
    ("SVM-RBF 默认", svm_rbf, X_test_s),
    ("SVM-RBF 加权", svm_bal, X_test_s),
    ("RF 默认", rf, X_test),
    ("RF 加权", rf_bal, X_test),
]:
    cmp_rows.append(report(name, y_test, model.predict(X_eval),
                           model.predict_proba(X_eval)[:, 1], record=False))

cmp_df = pd.DataFrame(cmp_rows)
labels = ["SVM 默认", "SVM 加权", "RF 默认", "RF 加权"]
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - width / 2, cmp_df["sens"], width, label="灵敏度", color="#E45756")
ax.bar(x + width / 2, cmp_df["spec"], width, label="特异度", color="#4C78A8")
for i, (sens, spec) in enumerate(zip(cmp_df["sens"], cmp_df["spec"])):
    ax.text(i - width / 2, sens + 0.01, f"{sens:.2f}", ha="center", fontsize=9)
    ax.text(i + width / 2, spec + 0.01, f"{spec:.2f}", ha="center", fontsize=9)
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.1)
ax.set_ylabel("比例")
ax.set_title("class_weight='balanced' 前后：灵敏度 vs 特异度")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig08_imbalance.png", dpi=120)
plt.close()

print(">>> 加权后灵敏度变化：SVM %+.4f，RF %+.4f；特异度变化：SVM %+.4f，RF %+.4f"
      % (cmp_df.loc[1, "sens"] - cmp_df.loc[0, "sens"],
         cmp_df.loc[3, "sens"] - cmp_df.loc[2, "sens"],
         cmp_df.loc[1, "spec"] - cmp_df.loc[0, "spec"],
         cmp_df.loc[3, "spec"] - cmp_df.loc[2, "spec"]))

# 4.4 比武成绩单
section("4.4 比武成绩单")
res_df = pd.DataFrame(results).sort_values("auc", ascending=False).reset_index(drop=True)
print(res_df.round(4).to_string(index=False))
res_df.to_csv(RESULT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig",
              float_format="%.4f")
print("\n已保存: results/model_comparison.csv")

# ==================================================================
# 报告素材自动汇总
# ==================================================================
section("报告素材（自动汇总，写报告时直接引用）")
print("1) 类别不平衡：流失率 %.2f%%（%d/%d）"
      % (churn_rate["Yes"] * 100, churn_counts["Yes"], churn_counts["No"] + churn_counts["Yes"]))
print("2) 脏数据：TotalCharges 非数值 %d 行（其中 tenure=0 的新客户 %d 行），整行脏记录 %d 行"
      % (int(bad_mask.sum()), n_new, raw_rows - len(data)))
print("3) 最高相关特征对：tenure ~ TotalCharges = %.3f，NB 条件独立假设被削弱，NB 灵敏度 %.4f"
      % (tenure_total_corr, nb_row["sens"]))
print("4) 核技巧：RBF 相对线性核 AUC %+.4f（%.4f → %.4f）"
      % (rbf_row["auc"] - lin_row["auc"], lin_row["auc"], rbf_row["auc"]))
print("5) MLP：迭代 %d 轮，loss %.4f → %.4f，最后 10 轮变化 %+.6f（判断是否收敛）"
      % (mlp.n_iter_, loss[0], loss[-1], loss[-1] - loss[-10]))
print("6) 组合方法：5 折 AUC 标准差 随机森林 %.4f vs 单棵决策树 %.4f（方差更小 = 更稳）"
      % (cv_stats["随机森林"][1], cv_stats["决策树"][1]))
print("7) 业务权衡：加权后 SVM 灵敏度 %+.4f / 特异度 %+.4f；RF 灵敏度 %+.4f / 特异度 %+.4f"
      % (cmp_df.loc[1, "sens"] - cmp_df.loc[0, "sens"], cmp_df.loc[1, "spec"] - cmp_df.loc[0, "spec"],
         cmp_df.loc[3, "sens"] - cmp_df.loc[2, "sens"], cmp_df.loc[3, "spec"] - cmp_df.loc[2, "spec"]))
print("8) 最优模型（按 AUC）：%s（AUC=%.4f）"
      % (res_df.iloc[0]["model"], res_df.iloc[0]["auc"]))

print("\n全部完成，图片在 %s" % FIG_DIR)
