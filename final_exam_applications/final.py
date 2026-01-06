import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    auc,
)

# ============================================================
# 1. DATA PREPARATION
# ============================================================
print("Veri Hazırlanıyor...")

df = pd.read_csv("pima_diabetes.csv")

zero_cols = ["glucose", "pressure", "triceps", "insulin", "mass", "pedigree", "age"]
valid_cols = [c for c in zero_cols if c in df.columns]
df[valid_cols] = df[valid_cols].replace(0, np.nan)
df[valid_cols] = df[valid_cols].fillna(df[valid_cols].median())

if df["diabetes"].dtype == "O":
    y = (df["diabetes"] == "pos").astype(int)
else:
    y = df["diabetes"]

X = df.drop("diabetes", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=239
)

# ============================================================
# 2. DECISION TREE & BAGGING & RANDOM FOREST ( GÖRSELLEŞTİRME)
# ============================================================
print("\n" + "=" * 60)
print("AĞAÇ BAZLI MODELLER VE GÖRSELLEŞTİRME")
print("=" * 60)

# --- Single Decision Tree ---
print("Decision Tree eğitiliyor ve çiziliyor...")
model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=239)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(20, 10))
plot_tree(
    model, feature_names=X.columns, class_names=["neg", "pos"], filled=True, fontsize=10
)
plt.title("Single Decision Tree (Max Depth=4)")
plt.show()


# --- Bagging ---
print("\nBagging eğitiliyor ve örnek bir ağaç çiziliyor...")
base_tree = DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=239)
bct = BaggingClassifier(
    estimator=base_tree, n_estimators=100, random_state=239, n_jobs=-1
)
bct.fit(X_train, y_train)
y_pred_bct = bct.predict(X_test)

# Bagging içinden ilk ağacı görselleştir
single_tree_bct = bct.estimators_[0]
plt.figure(figsize=(25, 15))  # Derin ağaçlar için boyut artırıldı
plot_tree(
    single_tree_bct,
    feature_names=X.columns,
    class_names=["neg", "pos"],
    filled=True,
    max_depth=4,
    fontsize=10,
)
plt.title("Tree from Bagging Classifier (First 4 levels shown)")
plt.show()


# --- Random Forest ---
print("\nRandom Forest eğitiliyor ve örnek bir ağaç çiziliyor...")
rfc_model = RandomForestClassifier(n_estimators=100, random_state=239, n_jobs=-1)
rfc_model.fit(X_train, y_train)
y_pred_rfc = rfc_model.predict(X_test)

# RF içinden ilk ağacı görselleştir
single_tree_rf = rfc_model.estimators_[0]
plt.figure(figsize=(25, 15))
plot_tree(
    single_tree_rf,
    feature_names=X.columns,
    class_names=["neg", "pos"],
    filled=True,
    max_depth=4,
    fontsize=10,
)
plt.title("Tree from Random Forest (First 4 levels shown)")
plt.show()


# ============================================================
# 3. LOGISTIC REGRESSION (Analiz + Model)
# ============================================================


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION - GELİŞMİŞ VARSAYIM TESTLERİ")
print("=" * 60)

# Veriyi Hazırlama (Scaling önemlidir)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_sm_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_train_sm_const = sm.add_constant(X_train_sm_df)

# ----------------------------------------------------------------
# 1. ÇOKLUBİRLİKTELİK (MULTICOLLINEARITY - VIF)
# ----------------------------------------------------------------

print("\n[1] ÇOKLUBİRLİKTELİK TESTİ (VIF)")
print("-" * 50)
vif_data = pd.DataFrame()
vif_data["feature"] = X_train_sm_const.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_train_sm_const.values, i)
    for i in range(X_train_sm_const.shape[1])
]

print(vif_data[vif_data["feature"] != "const"].round(2))
print("\n-> Yorum: VIF > 5-10 ise değişkenler arasında yüksek korelasyon vardır.")

# ----------------------------------------------------------------
# 2. MODEL İSTATİSTİKLERİ VE P-DEĞERLERİ (SIGNIFICANCE)
# ----------------------------------------------------------------
print("\n[2] MODEL KATSAYILARI VE P-DEĞERLERİ (STATSMODELS)")
print("-" * 50)
logit_model_sm = sm.Logit(y_train.reset_index(drop=True), X_train_sm_const)
result_sm = logit_model_sm.fit(disp=0)

print(result_sm.summary())
print(
    "\n-> Yorum: 'P>|z|' sütunu < 0.05 olan değişkenler istatistiksel olarak anlamlıdır."
)

# ----------------------------------------------------------------
# 3. DOĞRUSALLIK TESTİ (BOX-TIDWELL) - DÜZELTİLMİŞ
# ----------------------------------------------------------------

print("\n[3] DOĞRUSALLIK TESTİ (BOX-TIDWELL)")
print("-" * 50)
print("Hipotez: Değişken, logit (log-odds) ile doğrusal ilişkilidir.")

continuous_vars = [col for col in X_train.columns if X_train[col].nunique() > 10]

X_bt = X_train_sm_const.copy()

for col in continuous_vars:

    val_aligned = X_train[col].reset_index(drop=True)  
    shifted_val = val_aligned + abs(val_aligned.min()) + 1 
    X_bt[f"{col}_log_int"] = val_aligned * np.log(shifted_val) 
bt_model = sm.Logit(y_train.reset_index(drop=True), X_bt)

try:
    bt_result = bt_model.fit(disp=0)

    print(f"\n{'Değişken':<20} | {'P-Value':<10} | {'Durum'}")
    print("-" * 50)

    found_any = False
    for col in continuous_vars:
        interaction_col = f"{col}_log_int"
        if interaction_col in bt_result.pvalues:
            found_any = True
            p_val = bt_result.pvalues[interaction_col]
            status = "✓ Doğrusal" if p_val > 0.05 else "✗ Doğrusal DEĞİL"
            print(f"{col:<20} | {p_val:.4f}     | {status}")

    if not found_any:
        print(
            "Model yakınsamadığı için p-değerleri hesaplanamadı (Perfect Separation olabilir)."
        )

except Exception as e:
    print(f"Box-Tidwell hesaplanırken hata oluştu: {e}")
# ----------------------------------------------------------------
# 4. AYKIRI DEĞER ANALİZİ (COOK'S DISTANCE)
# ----------------------------------------------------------------
print("\n[4] ETKİLİ GÖZLEMLER VE AYKIRI DEĞERLER (COOK'S DISTANCE)")
print("-" * 50)
glm_model = sm.GLM(
    y_train.reset_index(drop=True), X_train_sm_const, family=sm.families.Binomial()
)
glm_results = glm_model.fit()
influence = glm_results.get_influence()
cooks = influence.cooks_distance[0]

plt.figure(figsize=(10, 5))
plt.stem(np.arange(len(cooks)), cooks, markerfmt=",")
plt.title("Cook's Distance (Etkili Gözlemler)")
plt.xlabel("Gözlem Indexi")
plt.ylabel("Cook's Distance")
threshold = 4 / len(X_train)
plt.axhline(
    y=threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.4f})"
)
plt.legend()
plt.show()

outliers = np.where(cooks > threshold)[0]
print(
    f"-> Eşik değerin ({threshold:.4f}) üzerinde {len(outliers)} adet etkili gözlem (outlier) tespit edildi."
)
print("-> Bu gözlemler modelin katsayılarını orantısız etkiliyor olabilir.")


log_reg_model = LogisticRegression(random_state=239, solver="liblinear", max_iter=1000)
log_reg_model.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg_model.predict(X_test_scaled)

# ============================================================
# 4. LDA & QDA
# ============================================================
print("\n" + "=" * 60)
print("LDA & QDA")
print("=" * 60)

# LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
y_pred_lda = lda_model.predict(X_test)

# QDA
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)
y_pred_qda = qda_model.predict(X_test)


print("\n" + "=" * 60)
print("LDA & QDA GÖRSELLEŞTİRME VE ANALİZİ")
print("=" * 60)

# ----------------------------------------------------------------
# 1. LDA PROJEKSİYONU (1 BOYUTLU AYRIM)
# ----------------------------------------------------------------

print("Grafik 1: LDA Projeksiyonu çiziliyor...")
X_lda_transformed = lda_model.transform(X_test)

plt.figure(figsize=(10, 6))
sns.histplot(
    x=X_lda_transformed.flatten(),
    hue=y_test,
    kde=True,
    element="step",
    palette={0: "orange", 1: "blue"},
)
plt.title("LDA Projeksiyonu: Sınıfların Ayrımı")
plt.xlabel("LD1 (Linear Discriminant 1)")
plt.ylabel("Frekans")
plt.legend(title="Diyabet", labels=["Pozitif", "Negatif"])
plt.grid(True, alpha=0.3)
plt.show()
print("QDA Olasılık Skorları Dağılımı çiziliyor...")

# 1. QDA modelinden pozitif sınıf (Diyabet=1) için olasılıkları al

y_prob_qda = qda_model.predict_proba(X_test)[:, 1]

viz_df = pd.DataFrame({"QDA_Positive_Probability": y_prob_qda, "True_Label": y_test})

plt.figure(figsize=(10, 6))

sns.histplot(
    data=viz_df,
    x="QDA_Positive_Probability",
    hue="True_Label",
    kde=True,  # 
    element="step", 
    stat="density", 
    palette={0: "orange", 1: "blue"}, 
    bins=20,  
)

plt.title("QDA Modeli: Diyabetli Olma Olasılığı (Positive Probability) Dağılımı")
plt.xlabel("Tahmin Edilen Olasılık (P(Diyabet | X))")
plt.ylabel("Yoğunluk (Density)")

plt.legend(
    title="Gerçek Durum",
    labels=["Pozitif (Diyabet)", "Negatif (Sağlıklı)"],
    loc="upper center",
)

plt.xlim(0, 1)  
plt.grid(True, alpha=0.3)
plt.show()
# ----------------------------------------------------------------
# 2. KARAR SINIRLARI KARŞILAŞTIRMASI (LDA vs QDA)
# ----------------------------------------------------------------

def plot_decision_boundary(X, y, model, title, ax):
    # Meshgrid oluşturma (Arka planı boyamak için)
    h = 0.5  # Adım büyüklüğü
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Tüm meshgrid noktaları için tahmin yap
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

   
    cmap_light = ListedColormap(["#FFD700", "#ADD8E6"]) 
    cmap_bold = ["darkorange", "darkblue"]  # Nokta renkleri

    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

    sns.scatterplot(
        x=X.iloc[:, 0],
        y=X.iloc[:, 1],
        hue=y,
        palette={0: "darkorange", 1: "darkblue"},
        alpha=0.8,
        edgecolor="black",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])

feature_cols = ["glucose", "mass"]
X_train_2d = X_train[feature_cols]
X_test_2d = X_test[feature_cols]


lda_2d = LinearDiscriminantAnalysis()
lda_2d.fit(X_train_2d, y_train)

qda_2d = QuadraticDiscriminantAnalysis()
qda_2d.fit(X_train_2d, y_train)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

plot_decision_boundary(X_test_2d, y_test, lda_2d, "LDA: Doğrusal Karar Sınırı", axes[0])
plot_decision_boundary(
    X_test_2d, y_test, qda_2d, "QDA: Karesel (Eğrisel) Karar Sınırı", axes[1]
)

plt.tight_layout()
plt.show()
# ============================================================
# 5. GRADIENT BOOSTING
# ============================================================
gbc_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=239
)
gbc_model.fit(X_train, y_train)
y_pred_gbc = gbc_model.predict(X_test)


print("\n" + "=" * 60)
print("GRADIENT BOOSTING - GÖRSELLEŞTİRME VE ANALİZ")
print("=" * 60)

feature_importance = gbc_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5

test_score = np.zeros((gbc_model.n_estimators,), dtype=np.float64)

for i, y_proba in enumerate(gbc_model.staged_predict_proba(X_test)):
    test_score[i] = log_loss(y_test, y_proba)

plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center", color="teal")
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.title("Feature Importance (Değişken Önemi)")
plt.xlabel("Önem Derecesi (Importance Score)")
plt.grid(True, alpha=0.3)


plt.subplot(1, 2, 2)
plt.plot(
    np.arange(gbc_model.n_estimators) + 1,
    gbc_model.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(gbc_model.n_estimators) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations (Ağaç Sayısı)")
plt.ylabel("Deviance (Log Loss)")
plt.title("Gradient Boosting - Öğrenme Eğrisi")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 6. MODEL KARŞILAŞTIRMASI VE GÖRSELLEŞTİRME
# ============================================================


def evaluate_model(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "Sensitivity": sens,
        "Specificity": spec,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
    }


dt_prob = model.predict_proba(X_test)[:, 1]
bct_prob = bct.predict_proba(X_test)[:, 1]
rfc_prob = rfc_model.predict_proba(X_test)[:, 1]
log_prob = log_reg_model.predict_proba(X_test_scaled)[:, 1]  
lda_prob = lda_model.predict_proba(X_test)[:, 1]
qda_prob = qda_model.predict_proba(X_test)[:, 1]
gbc_prob = gbc_model.predict_proba(X_test)[:, 1]


models_dict = {
    "Decision Tree": (y_pred, dt_prob),
    "Bagging": (y_pred_bct, bct_prob),
    "Random Forest": (y_pred_rfc, rfc_prob),
    "Logistic Reg": (y_pred_log_reg, log_prob),
    "LDA": (y_pred_lda, lda_prob),
    "QDA": (y_pred_qda, qda_prob),
    "Gradient Boost": (y_pred_gbc, gbc_prob),
}

results = {}
for name, (preds, probs) in models_dict.items():
    results[name] = evaluate_model(y_test, preds, probs)

results_df = pd.DataFrame.from_dict(results, orient="index")


print("\nROC Eğrileri çiziliyor...")
plt.figure(figsize=(12, 8))
for name, (preds, probs) in models_dict.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()


print("\n" + "=" * 60)
print("MODEL PERFORMANS KARŞILAŞTIRMASI")
print("=" * 60)
print(results_df.round(4))

print("\nÖZET:")
print(
    f"En yüksek Accuracy: {results_df['Accuracy'].idxmax()} ({results_df['Accuracy'].max():.4f})"
)
print(f"En yüksek AUC: {results_df['AUC'].idxmax()} ({results_df['AUC'].max():.4f})")
