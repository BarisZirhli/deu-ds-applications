import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan


######################################################################
#                                                                    #
#                              1                                     #
#                                                                    #
#                                                                    #
######################################################################

SEED = 2223915239

# Veri setini yükle
url = "https://www.statlearning.com/s/Credit.csv"
df = pd.read_csv(url)
print(" Veri setinin ilk 5 satırı:")
print(df.head())

print("\n Tanımlayıcı istatistikler:")
desc = df.describe(include="all")
print(desc)

# Histogramlarla görselleştirme
numeric_cols = df.select_dtypes(include=["number"]).columns
print(numeric_cols)
df[numeric_cols].hist(figsize=(12, 10), bins=15, edgecolor="black")
plt.suptitle("Histogramlar", fontsize=16)
# plt.show()

# Boxplotlar ile aykırı gözlem kontrolü
plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, len(numeric_cols), i + 1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
# plt.show()


######################################################################
#                                                                    #
#                                 2                                  #
#                                                                    #
#                                                                    #
######################################################################


sns.pairplot(df[numeric_cols])
plt.suptitle("Matris Plot", y=1.02)
# plt.show()

corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Korelasyon Matrisi")
# plt.show()

# Dummies
obj_cols = df.select_dtypes(include=["object"]).columns
print("\n🔹 Kategorik değişkenler:")
print(obj_cols)
df_dum = pd.get_dummies(df, columns=obj_cols, drop_first=True)

X = df_dum.drop("Balance", axis=1)
y = df_dum["Balance"]


######################################################################
#                                                                    #
#                               3                                    #
#                                                                    #
#                                                                    #
######################################################################


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED
)

# Boolean sütunları int yap
for col in X_train.columns:
    if X_train[col].dtype == "bool":
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)

# Tüm sütunları float'a çevir
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Intercept ekle
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Modeli oluştur
model = sm.OLS(y_train, X_train_sm).fit()
print("\n🔹 Model Özeti:")
print(model.summary())


######################################################################
#                                                                    #
#                                4                                   #
#                                                                    #
#                                                                    #
######################################################################


# Önce p-value < 0.05 olanları belirle
summary_df = pd.DataFrame({"Coefficient": model.params, "P-value": model.pvalues})
significant_vars = summary_df[summary_df["P-value"] < 0.05]
print(significant_vars)
sig_cols = significant_vars.index.drop("const")

# Sadece anlamlı değişkenler
X_train_sig = X_train[sig_cols].copy()
X_test_sig = X_test[sig_cols].copy()

# Intercept ekle
X_train_sig_sm = sm.add_constant(X_train_sig)
X_test_sig_sm = sm.add_constant(X_test_sig)

# Modeli oluştur
model_sig = sm.OLS(y_train, X_train_sig_sm).fit()
print("\n Significant değişkenlerle Model Özeti:")
print(model_sig.summary())


######################################################################
#                                                                    #
#                               5                                    #
#                                                                    #
#                                                                    #
######################################################################


X_train_sig = X_train[significant_vars.index.drop("const")]
vif_df = pd.DataFrame()
vif_df["feature"] = X_train_sig.columns
vif_df["VIF"] = [
    variance_inflation_factor(X_train_sig.values, i)
    for i in range(X_train_sig.shape[1])
]
print("\n VIF Değerleri:")
print(vif_df)


######################################################################
#                                                                    #
#                               6                                    #
#                                                                    #
#                                                                    #
######################################################################


linreg = LinearRegression()
X_sig_all = X_train_sig.copy()

efs = EFS(
    linreg,
    min_features=1,
    max_features=min(10, X_sig_all.shape[1]),
    scoring="r2",
    cv=5,
    print_progress=True,
)
efs = efs.fit(X_sig_all, y_train)
results_df = pd.DataFrame(efs.get_metric_dict()).T.sort_values(
    "avg_score", ascending=False
)
print("\n İlk 2 en iyi alt küme modeli:")
print(results_df.head(2)[["feature_names", "avg_score"]])


######################################################################
#                                                                    #
#                               7                                    #
#                                                                    #
#                                                                    #
######################################################################


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    press = np.sum((y_true - y_pred) ** 2)
    return press, rmse, mae


metrics = {}

# Ana model
y_pred_main = model.predict(X_test_sm)
metrics["Ana Model"] = calc_metrics(y_test, y_pred_main)

# Alternatif 2 model
for i in range(2):
    features = list(results_df.iloc[i]["feature_names"])
    X_train_tmp = sm.add_constant(X_train[features])
    X_test_tmp = sm.add_constant(X_test[features])
    model_tmp = sm.OLS(y_train, X_train_tmp).fit()
    y_pred_tmp = model_tmp.predict(X_test_tmp)
    metrics[f"EFS Model {i+1}"] = calc_metrics(y_test, y_pred_tmp)

metrics_df = pd.DataFrame(metrics, index=["PRESS", "RMSE", "MAE"]).T
print("\n Test Seti Performans Metrikleri:")
print(metrics_df)

best_model_name = metrics_df.mean(axis=1).idxmin()
print(f"\n🔹 En uygun model: {best_model_name}")


best_model_features = list(
    results_df.iloc[1]["feature_names"]
)  # örnek olarak 1. EFS modeli
X_best_train = sm.add_constant(X_train[best_model_features])
X_best_test = sm.add_constant(X_test[best_model_features])
model_best = sm.OLS(y_train, X_best_train).fit()


######################################################################
#                                                                    #
#                               8                                    #
#                                                                    #
#                                                                    #
######################################################################


residuals = model_best.resid
# 1. Histogram ve Q-Q plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Histogram
sns.histplot(residuals, kde=True, ax=axes[0])
axes[0].axvline(
    residuals.mean(),
    color="red",
    linestyle="--",
    label=f"Ortalama: {residuals.mean():.2f}",
)
axes[0].set_xlabel("Residuallar")
axes[0].set_ylabel("Frekans")
axes[0].set_title("Residual Histogramı ve KDE")
axes[0].legend()

# Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot - Normal Dağılım Kontrolü")
axes[1].set_xlabel("Teorik Quantilelar")
axes[1].set_ylabel("Örneklem Quantileları")

plt.tight_layout()
plt.show()


# Shapiro-Wilk testi
shapiro_test = stats.shapiro(residuals)
print("\n Shapiro-Wilk Test Sonuçları:")
print(f"   Test İstatistiği: {shapiro_test.statistic:.6f}")
print(f"   P-değeri: {shapiro_test.pvalue:.6f}")


######################################################################
#                                                                    #
#                               9                                    #
#                                                                    #
#                                                                    #
######################################################################

# Residual vs Fitted plot
fitted_values = model_best.fittedvalues
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Residual vs Fitted
axes[0].scatter(fitted_values, residuals, alpha=0.6)
axes[0].axhline(y=0, color="red", linestyle="--", linewidth=2)
axes[0].set_xlabel("Tahmin Edilen Değerler (Fitted)")
axes[0].set_ylabel("Residuallar")
axes[0].set_title("Residual vs Fitted Plot\n(Sabit Varyans Kontrolü)")

# Residual'ların mutlak değeri vs Fitted (daha iyi görselleştirme)
axes[1].scatter(fitted_values, np.abs(residuals), alpha=0.6)
axes[1].set_xlabel("Tahmin Edilen Değerler (Fitted)")
axes[1].set_ylabel("|Residuallar|")
axes[1].set_title("Mutlak Residual vs Fitted Plot\n(Varyans Pattern Kontrolü)")

plt.tight_layout()
plt.show()

# Breusch-Pagan testi
bp_test = het_breuschpagan(residuals, model_best.model.exog)
labels = ["LM İstatistiği", "LM p-değeri", "F-istatistiği", "F p-değeri"]
bp_results = dict(zip(labels, bp_test))


print("\n" + "=" * 60)
print("\n Breusch-Pagan Test Sonuçları:")
for key, value in bp_results.items():
    print(f"   {key}: {value:.6f}")


print(" UÇ DEĞER VE ETKİN GÖZLEM ANALİZİ")
print("=" * 60)

######################################################################
#                                                                    #
#                               10                                   #
#                                                                    #
#                                                                    #
######################################################################


influence = model_best.get_influence()
student_resid = influence.resid_studentized_internal

outlier_threshold = 3
outliers_resid = np.where(np.abs(student_resid) > outlier_threshold)[0]

print(f"\n STUDENTIZED RESIDUALS İLE AYKIRI DEĞERLER")
print(f"   Eşik değer: ±{outlier_threshold}")
print(f"   Tespit edilen aykırı değer sayısı: {len(outliers_resid)}")

if len(outliers_resid) > 0:
    print(f"   Aykırı gözlem indeksleri: {outliers_resid}")
    for idx in outliers_resid:
        print(f"      Gözlem {idx}: Studentized Residual = {student_resid[idx]:.3f}")
else:
    print("    Studentized residuals'a göre aykırı değer bulunamadı")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(range(len(student_resid)), student_resid, alpha=0.6)
plt.axhline(
    y=outlier_threshold, color="r", linestyle="--", label=f"+{outlier_threshold} sigma"
)
plt.axhline(
    y=-outlier_threshold, color="r", linestyle="--", label=f"-{outlier_threshold} sigma"
)
plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
plt.xlabel("Gözlem İndeksi")
plt.ylabel("Studentized Residuals")
plt.title("Studentized Residuals - Aykırı Değer Kontrolü")
plt.legend()
plt.grid(True, alpha=0.3)

# Histogram
plt.subplot(1, 2, 2)
sns.histplot(student_resid, kde=True)
plt.axvline(outlier_threshold, color="r", linestyle="--", label=f"±{outlier_threshold}")
plt.axvline(-outlier_threshold, color="r", linestyle="--")
plt.xlabel("Studentized Residuals")
plt.ylabel("Frekans")
plt.title("Studentized Residuals Dağılımı")
plt.legend()

plt.tight_layout()
plt.show()

# Cook's distance
cooks_d = influence.cooks_distance[0]

# Cook's distance eşik değeri (4/n)
cook_threshold = 4 / len(cooks_d)
influential_cook = np.where(cooks_d > cook_threshold)[0]

print(f"\n COOK'S DISTANCE İLE ETKİLİ GÖZLEMLER")
print(f"   Eşik değer: {cook_threshold:.4f} (4/n)")
print(f"   Tespit edilen etkili gözlem sayısı: {len(influential_cook)}")


# Grafik
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.stem(cooks_d, markerfmt=",", basefmt=" ")
plt.axhline(
    y=cook_threshold, color="r", linestyle="--", label=f"Eşik: {cook_threshold:.4f}"
)
plt.xlabel("Gözlem İndeksi")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance - Etkili Gözlemler")
plt.legend()
plt.grid(True, alpha=0.3)


plt.subplot(1, 2, 2)
plt.scatter(model_best.fittedvalues, cooks_d, alpha=0.6)
plt.axhline(
    y=cook_threshold, color="r", linestyle="--", label=f"Eşik: {cook_threshold:.4f}"
)
plt.xlabel("Tahmin Edilen Değerler")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance vs Fitted Values")
plt.legend()

plt.tight_layout()
plt.show()

leverage = influence.hat_matrix_diag

k = model_best.df_model 
leverage_threshold = 2 * (k + 1) / len(leverage)
high_leverage = np.where(leverage > leverage_threshold)[0]

print(f"\n LEVERAGE DEĞERLERİ İLE ETKİLİ GÖZLEMLER")
print(f"   Bağımsız değişken sayısı (k): {k}")
print(f"   Eşik değer: {leverage_threshold:.4f} (2*(k+1)/n)")
print(f"   Yüksek leveragelı gözlem sayısı: {len(high_leverage)}")


# Grafik
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.stem(leverage, markerfmt=",", basefmt=" ")
plt.axhline(
    y=leverage_threshold,
    color="r",
    linestyle="--",
    label=f"Eşik: {leverage_threshold:.4f}",
)
plt.xlabel("Gözlem İndeksi")
plt.ylabel("Leverage")
plt.title("Leverage Değerleri - Etkili Gözlemler")
plt.legend()
plt.grid(True, alpha=0.3)


plt.subplot(1, 2, 2)
plt.scatter(leverage, student_resid, alpha=0.6)
plt.axhline(
    y=outlier_threshold, color="r", linestyle="--", alpha=0.7, label="Residual eşik"
)
plt.axhline(y=-outlier_threshold, color="r", linestyle="--", alpha=0.7)
plt.axvline(
    x=leverage_threshold, color="g", linestyle="--", alpha=0.7, label="Leverage eşik"
)
plt.xlabel("Leverage")
plt.ylabel("Studentized Residuals")
plt.title("Leverage vs Studentized Residuals")
plt.legend()

plt.tight_layout()
plt.show()


dffits = influence.dffits[0]
dffits_threshold = 2 * np.sqrt((k + 1) / len(dffits))
high_dffits = np.where(np.abs(dffits) > dffits_threshold)[0]

print(f"\nDFFITS İLE MODEL TAHMİNLERİNE ETKİ")
print(f"   Eşik değer: {dffits_threshold:.4f}")
print(f"   Yüksek DFFITS'lı gözlem sayısı: {len(high_dffits)}")

if len(high_dffits) > 0:
    print(f"   Yüksek DFFITS'lı gözlemler: {high_dffits}")
    for idx in high_dffits[:5]:  # İlk 5 tanesini göster
        print(f"      Gözlem {idx}: DFFITS = {dffits[idx]:.4f}")


dfbetas = influence.dfbetas
dfbetas_threshold = 2 / np.sqrt(len(dfbetas))
high_dfbetas = {}

for i, var_name in enumerate(model_best.model.exog_names):
    high_indices = np.where(np.abs(dfbetas[:, i]) > dfbetas_threshold)[0]
    if len(high_indices) > 0:
        high_dfbetas[var_name] = high_indices

print(f"\n DFBETAS İLE KATSAYILARA ETKİ")
print(f"   Eşik değer: {dfbetas_threshold:.4f}")
if high_dfbetas:
    print("   Katsayıları etkileyen gözlemler:")
    for var, indices in high_dfbetas.items():
        print(f"      {var}: {indices} gözlem(ler)")
else:
    print(" Hiçbir katsayıyı önemli derecede etkileyen gözlem bulunamadı")


# Tüm etkili gözlemleri birleştir
all_influential = set(outliers_resid) | set(influential_cook) | set(high_leverage) | set(high_dffits)

print(f"\n ETKİLİ GÖZLEMLERİN ÖZETİ")
print("="*50)
print(f"Toplam gözlem sayısı: {len(student_resid)}")
print(f"Toplam etkili gözlem sayısı: {len(all_influential)}")
print(f"Etkili gözlem yüzdesi: {len(all_influential)/len(student_resid)*100:.1f}%")

if all_influential:
    print(f"\n Etkili gözlem indeksleri: {sorted(all_influential)}")
    
    # Etkili gözlemlerin detaylı analizi
    print(f"\n ETKİLİ GÖZLEMLERİN DETAYLI ANALİZİ")
    print("-" * 60)
    print(f"{'Gözlem':<8} {'Stud.Resid':<12} {'Cooks D':<12} {'Leverage':<12} {'DFFITS':<12}")
    print("-" * 60)
    
    for idx in sorted(all_influential):
        print(f"{idx:<8} {student_resid[idx]:<12.3f} {cooks_d[idx]:<12.4f} "
              f"{leverage[idx]:<12.4f} {dffits[idx]:<12.4f}")
else:
    print("\n TEBRİKLER! Modelde etkili gözlem/aykırı değer bulunamadı.")
if all_influential:
    print(f"\n ETKİLİ GÖZLEMLERİN MODEL PERFORMANSINA ETKİSİ")

    clean_indices = [i for i in range(len(student_resid)) if i not in all_influential]

    if len(clean_indices) > len(model_best.model.exog_names):  
  
        X_clean = model_best.model.exog[clean_indices]
        y_clean = model_best.model.endog[clean_indices]

   
    try:
        model_clean = sm.OLS(y_clean, X_clean).fit()
        
        print(f"Orijinal model R²: {model_best.rsquared:.4f}")
        print(f"Temiz model R²: {model_clean.rsquared:.4f}")
        print(f"Değişim: {model_clean.rsquared - model_best.rsquared:+.4f}")

        # Katsayı değişimleri 
        print(f"\n KATSAYI DEĞİŞİMLERİ")
        print(f"{'Değişken':<15} {'Orijinal':<10} {'Temiz':<10} {'Değişim':<10} {'Değişim %':<12}")
        print("-" * 60)
        
        for i, var in enumerate(model_best.model.exog_names):
            # Orijinal modelden katsayıları al
            orig_coef = model_best.params.iloc[i]
            
           
            clean_coef = model_clean.params[i]  
            
            change = clean_coef - orig_coef
            change_pct = (change / orig_coef) * 100 if orig_coef != 0 else 0
            print(f"{var:15} {orig_coef:>9.3f} {clean_coef:>9.3f} {change:>9.3f} {change_pct:>10.1f}%")
            
    except Exception as e:
        print(f"❌ Model oluşturma hatası: {e}")
        print(f"X_clean shape: {X_clean.shape}")
        print(f"y_clean shape: {y_clean.shape}")

######################################################################
#                                                                    #
#                               11                                   #
#                                                                    #
#                                                                    #
######################################################################


model_vars = ["const", "Income", "Limit", "Cards", "Student_Yes"]
new_observation = [1.0, 50.0, 5000.0, 3.0, 1.0]

tahmin_sonucu = model_best.get_prediction([new_observation])
sonuc = tahmin_sonucu.summary_frame(alpha=0.05)

print("YENİ MÜŞTERİ TAHMİNİ")
print("=" * 30)
print(
    f"%95 Güven Aralığı: {sonuc['mean_ci_lower'].values[0]:.0f} - {sonuc['mean_ci_upper'].values[0]:.0f} TL"
)
print(
    f"%95 Kestirim Aralığı: {sonuc['obs_ci_lower'].values[0]:.0f} - {sonuc['obs_ci_upper'].values[0]:.0f} TL"
)


print("• Ortalama müşteri bakiyesi: ~{:.0f} TL".format(sonuc["mean"].values[0]))
print(
    "• Tek bir müşteri bakiyesi: {:.0f} - {:.0f} TL arası".format(
        sonuc["obs_ci_lower"].values[0], sonuc["obs_ci_upper"].values[0]
    )
)

X_clean = model_best.model.exog[clean_indices]
y_clean = model_best.model.endog[clean_indices]


model_robust = sm.OLS(y_clean, X_clean).fit(cov_type="HC3")
print("ROBUST STANDART HATALARLI FİNAL MODEL")
print("=" * 50)
print(model_robust.summary())

print("\n STANDART HATA KARŞILAŞTIRMASI")
print("Değişken\t\t OLS SE\t\t Robust SE\t Değişim %")
print("-" * 60)


ols_se_array = model_best.bse.values
robust_se_array = model_robust.bse

for i in range(len(ols_se_array)):
    var_name = model_best.model.exog_names[i]
    ols_se = ols_se_array[i]
    robust_se = robust_se_array[i]
    change_pct = ((robust_se - ols_se) / ols_se) * 100 if ols_se != 0 else 0

    print(f"{var_name:15} {ols_se:10.3f} {robust_se:10.3f} {change_pct:10.1f}%")

# Yeni gözlem tahmini - Robust model ile

model_vars = ["const", "Income", "Limit", "Cards", "Student_Yes"]
new_observation = [1.0, 50.0, 5000.0, 3.0, 1.0]


tahmin_robust = model_robust.get_prediction([new_observation])
sonuc_robust = tahmin_robust.summary_frame(alpha=0.05)

print("\n YENİ MÜŞTERİ TAHMİNİ - ROBUST MODEL")
print("=" * 40)
print(
    f"%95 Güven Aralığı: {sonuc_robust['mean_ci_lower'].values[0]:.0f} - {sonuc_robust['mean_ci_upper'].values[0]:.0f} TL"
)
print(
    f"%95 Kestirim Aralığı: {sonuc_robust['obs_ci_lower'].values[0]:.0f} - {sonuc_robust['obs_ci_upper'].values[0]:.0f} TL"
)
