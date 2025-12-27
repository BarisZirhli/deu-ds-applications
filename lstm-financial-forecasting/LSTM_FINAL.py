import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import warnings
import matplotlib.pyplot as plt
from datetime import timedelta

warnings.filterwarnings("ignore")
plt.style.use("ggplot")

# ============================================================
# 1) YAHOO VERİSİ — BIST100 USD FİYATI ve Türetilmiş Özellikler
# ============================================================

print(" Yahoo Finance'tan veri çekiliyor...")
start_date = "2015-01-01"
bist = yf.download("XU100.IS", start=start_date, interval="1d", progress=False)
usdtry = yf.download("TRY=X", start=start_date, interval="1d", progress=False)

df = pd.DataFrame()
df["Close"] = bist["Close"]
df["High"] = bist["High"]
df["Low"] = bist["Low"]
df["Volume"] = bist["Volume"]
df["USDTRY"] = usdtry["Close"]
df = df.dropna()

# USD bazlı fiyat (TARGET)
df["Close_USD"] = df["Close"] / df["USDTRY"]

print(" Veri alındı:", df.shape)

# --- TÜRETİLMİŞ FEATURE ÜRETİMİ ---
df["returns"] = df["Close_USD"].pct_change()
df["log_returns"] = np.log1p(df["returns"])
df["rolling_mean_10"] = df["Close_USD"].rolling(10).mean()
df["rolling_mean_50"] = df["Close_USD"].rolling(50).mean()
df["rolling_std_10"] = df["Close_USD"].rolling(10).std()
df["high_low_range"] = df["High"] - df["Low"]
df["volume_change"] = df["Volume"].pct_change()
ema_fast = df["Close_USD"].ewm(span=12, adjust=False).mean()
ema_slow = df["Close_USD"].ewm(span=26, adjust=False).mean()
df["macd"] = ema_fast - ema_slow
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"] = df["macd"] - df["macd_signal"]


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


df["rsi14"] = compute_rsi(df["Close_USD"], 14)
df = df.dropna().reset_index()

# ============================================================
# 2) KORELASYON — EN İYİ 3 FEATURE SEÇİMİ
# ============================================================

target_col = "Close_USD"

corr = df.corr()[target_col].abs().sort_values(ascending=False)
corr = corr.drop(target_col)

highly_correlated = [
    f for f in corr.index.tolist() if f not in ["Date", "Close", "High", "Low"]
]
top_features = [target_col] + highly_correlated[0:3]

print(f"\n En iyi 4 feature (Close_USD dahil): {top_features}")

# ============================================================
# 3) MULTIVARIATE DATASET OLUŞTURMA
# ============================================================

feature_data = df[top_features].values
target = df[target_col].values.reshape(-1, 1)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaled_x = scaler_x.fit_transform(feature_data)
scaled_y = scaler_y.fit_transform(target)
time_step = 20


def create_dataset(X, y, time_step=25):
    xs, ys = [], []
    for i in range(len(X) - time_step):
        xs.append(X[i : i + time_step])
        ys.append(y[i + time_step])
    return np.array(xs), np.array(ys)


X_lstm, y_lstm = create_dataset(scaled_x, scaled_y, time_step)
test_ratio = 0.25
test_size = int(len(X_lstm) * test_ratio)

X_train, X_test = X_lstm[:-test_size], X_lstm[-test_size:]
y_train, y_test = y_lstm[:-test_size], y_lstm[-test_size:]

print(f"Eğitim/Test Veri Boyutları: X_train:{X_train.shape}, X_test:{X_test.shape}")

# ============================================================
# 4) LSTM MODELİ TANIMI VE EĞİTİMİ
# ============================================================

model = Sequential(
    [
        LSTM(
            128,
            return_sequences=False,
            input_shape=(time_step, len(top_features)),
            kernel_regularizer=regularizers.l2(0.001),
        ),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.3),
        Dense(1),
    ]
)

early = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=0.00001)
model.compile(optimizer="adam", loss="mse")


print("\n Model eğitiliyor...")

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    shuffle=False,
    callbacks=[early, reduce],
    verbose=1,
)

# ============================================================
# 5) PERFORMANS RAPORU (Hazırlık)
# ============================================================
print(model.summary())
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Tahminleri ters ölçekleme
train_pred_original = scaler_y.inverse_transform(train_pred)
test_pred_original = scaler_y.inverse_transform(test_pred)

y_train_original = scaler_y.inverse_transform(y_train)
y_test_original = scaler_y.inverse_transform(y_test)

train_r2 = r2_score(y_train_original, train_pred_original)
test_r2 = r2_score(y_test_original, test_pred_original)

train_mae = mean_absolute_error(y_train_original, train_pred_original)
test_mae = mean_absolute_error(y_test_original, test_pred_original)

# ============================================================
# 6) 30 GÜNLÜK TAHMİN — AUTOREGRESSIVE
# ============================================================

last_seq = X_lstm[-1].copy()
future_predictions_scaled = []

for i in range(30):
    pred_scaled = model.predict(
        last_seq.reshape(1, time_step, len(top_features)), verbose=0
    )[0, 0]
    future_predictions_scaled.append(pred_scaled)

    new_row = last_seq[-1].copy()
    new_row[0] = pred_scaled  # Close_USD'yi geri besle

    last_seq = np.vstack([last_seq[1:], new_row])

# Tahminleri ters ölçekleme
future_predictions = scaler_y.inverse_transform(
    np.array(future_predictions_scaled).reshape(-1, 1)
).flatten()

# ============================================================
# 7) GÖRSELLEŞTİRME VE RAPORLAMA İÇİN DEĞİŞKEN TANIMLARI
# ============================================================

df_recent = df[["Date", "Close_USD"]].iloc[-len(X_lstm) :].reset_index(drop=True)
original_series = df_recent["Close_USD"].values
df_recent = df_recent.iloc[-60:].reset_index(drop=True)  # Son 60 gün

last_date = df_recent["Date"].iloc[-1]
last_actual = df_recent["Close_USD"].iloc[-1]

# Gelecek Tarihleri Oluşturma
future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
future_dates = pd.to_datetime(future_dates)


series_transformed = np.log(df["Close_USD"].values)
standard_scaler = StandardScaler()
scaled_series = standard_scaler.fit_transform(df["Close_USD"].values.reshape(-1, 1))

# ============================================================
# 8) GÖRSELLEŞTİRME (Tüm Paneli)
# ============================================================

plt.figure(figsize=(15, 12))

# --- 1. Ana Tahmin Grafiği: Geçmiş ve Gelecek Trendi ---

plt.subplot(3, 2, 1)
recent_dates_plot = df_recent["Date"]
recent_values_plot = df_recent["Close_USD"]

plt.plot(
    recent_dates_plot,
    recent_values_plot,
    label="Son 60 Gün (Gerçek)",
    color="blue",
    linewidth=2,
)
plt.plot(
    future_dates,
    future_predictions,
    label="30 Gün Tahmin",
    color="red",
    linewidth=2,
    marker="o",
    markersize=3,
)

plt.axvline(x=last_date, color="green", linestyle="--", label="Son Gün", alpha=0.7)
plt.title("1. ANA TAHMİN GRAFİĞİ: GEÇMİŞ VE GELECEK", fontweight="bold", fontsize=12)
plt.xlabel("Tarih")
plt.ylabel("USD Fiyatı")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# --- 2. Logaritmik Dönüşüm Dağılımı ---

plt.subplot(3, 2, 2)
plt.hist(
    series_transformed,
    bins=50,
    alpha=0.7,
    color="skyblue",
    edgecolor="black",
    density=True,
)
plt.axvline(
    series_transformed.mean(),
    color="red",
    linestyle="--",
    label=f"Ortalama: {series_transformed.mean():.3f}",
)
plt.title("2. LOGARİTMİK DÖNÜŞÜM DAĞILIMI", fontweight="bold", fontsize=12)
plt.xlabel("Logaritmik Değer")
plt.ylabel("Yoğunluk")
plt.legend()
plt.grid(True, alpha=0.3)

# --- 3. Standard Scaled Dağılım ---

plt.subplot(3, 2, 3)
plt.hist(
    scaled_series.flatten(),
    bins=50,
    alpha=0.7,
    color="lightgreen",
    edgecolor="black",
    density=True,
)
plt.axvline(
    scaled_series.mean(),
    color="red",
    linestyle="--",
    label=f"Ortalama: {scaled_series.mean():.3f}",
)
plt.title("3. STANDARD SCALED DAĞILIMI", fontweight="bold", fontsize=12)
plt.xlabel("Scaled Değer")
plt.ylabel("Yoğunluk")
plt.legend()
plt.grid(True, alpha=0.3)

# --- 4. Günlük Yüzde Değişimler ---

plt.subplot(3, 2, 4)
daily_changes = np.diff(future_predictions) / future_predictions[:-1] * 100

plt.bar(
    range(1, 30),
    daily_changes,
    color=["green" if x > 0 else "red" for x in daily_changes],
    alpha=0.7,
)
plt.axhline(y=0, color="black", linewidth=0.5)
plt.title("4. GÜNLÜK TAHMİN DEĞİŞİMLERİ (%)", fontweight="bold", fontsize=12)
plt.xlabel("Gün")
plt.ylabel("Değişim %")
plt.grid(True, alpha=0.3)

# --- 5. Model Eğitim Kaybı (Loss) ---

plt.subplot(3, 2, 5)
plt.plot(history.history["loss"], label="Eğitim Kaybı", linewidth=2)
plt.plot(history.history["val_loss"], label="Doğrulama Kaybı", linewidth=2)
plt.title(
    "5. MODEL EĞİTİM KAYBI (LOSS) - Stabilite Kontrolü", fontweight="bold", fontsize=12
)
plt.ylabel("MSE")
plt.xlabel("Epok")
plt.legend()
plt.grid(True, alpha=0.3)

# --- 6. Gerçek vs Tahmin Dağılımı (R² Kontrolü)

plt.subplot(3, 2, 6)
plt.scatter(
    y_test_original.flatten(),
    test_pred_original.flatten(),
    alpha=0.6,
    label=f"Test Verisi (R²: {test_r2:.4f})",
)
max_val = max(y_test_original.max(), test_pred_original.max())
plt.plot([0, max_val], [0, max_val], "r--", label="İdeal Tahmin (y=x)")

plt.title(
    "6. TAHMİN vs GERÇEK (Test Verisi) - R² Kontrolü", fontweight="bold", fontsize=12
)
plt.xlabel("Gerçek Değer")
plt.ylabel("Tahmin Değer")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 9) PERFORMANS RAPORU (Ekstra Raporlama)
# ============================================================

print(f"\n PERFORMANS RAPORU (TÜM VERİ SETİ ÜZERİNDEN):")
print("=" * 50)
print(f"Eğitim MAE: {train_mae:.2f}")
print(f"Eğitim R²: {train_r2:.4f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Test R²: {test_r2:.4f}")

print(f"\n 30 GÜNLÜK TAHMİN ÖZETİ:")
print(f"Başlangıç Değeri: {last_actual:.2f}")
print(f"30 Gün Sonu Tahmin: {future_predictions[-1]:.2f}")
total_change_pct = (future_predictions[-1] - last_actual) / last_actual * 100
print(f"Toplam Değişim: {total_change_pct:+.2f}%")
print(
    f"Tahmin Volatilitesi: {np.std(np.diff(future_predictions) / future_predictions[:-1] * 100):.2f}% (Günlük % Değişim Standard Sapması)"
)

print(
    f"\n DÜZENLİLEŞTİRİLMİŞ MODEL ÇALIŞTIRMA VE GÖRSELLEŞTİRME SİSTEMİ BAŞARIYLA TAMAMLANDI."
)
# ============================================================

# Tahmin ve Gerçek Değerleri tek bir diziye birleştirme
# full_actual dizisi y_train_original ve y_test_original'ın birleşimidir
full_actual = np.concatenate((y_train_original, y_test_original), axis=0).flatten()

# full_pred dizisi train_pred_original ve test_pred_original'ın birleşimidir
full_pred = np.concatenate((train_pred_original, test_pred_original), axis=0).flatten()

# Grafikte kullanılacak zaman serisi (X_lstm uzunluğunda)
# Bu DataFrame, tahmin yapılan dönemin tamamını kapsar
df_plot = df[["Date", "Close_USD"]].iloc[-len(X_lstm) :].reset_index(drop=True)

train_test_split_date = df_plot["Date"].iloc[len(y_train_original)]
r2_diff = train_r2 - test_r2

plt.figure(figsize=(12, 6))

# --- Tahmin Kıyaslaması Grafiği ---

# 1. Gerçek Değerler (Gri Çizgi)
plt.plot(
    df_plot["Date"], full_actual, label="Gerçek Değerler", color="gray", linewidth=2
)

# 2. Eğitim Tahmini (Mavi Çizgi)
plt.plot(
    df_plot["Date"].iloc[: len(y_train_original)],
    train_pred_original,
    label=f"Eğitim Tahmini (R²: {train_r2:.4f})",
    color="blue",
    linewidth=2,
)

# 3. Test Tahmini (Kırmızı Çizgi)
plt.plot(
    df_plot["Date"].iloc[len(y_train_original) :],
    test_pred_original,
    label=f"Test Tahmini (R²: {test_r2:.4f})",
    color="red",
    linewidth=2,
)

# 4. Eğitim/Test Sınırı (Yeşil Kesikli Çizgi)
plt.axvline(
    x=train_test_split_date,
    color="green",
    linestyle="--",
    label="Eğitim/Test Sınırı",
    alpha=0.8,
)

plt.title(
    f"LSTM Tahmin Kıyaslaması (Eğitim vs. Test R² Farkı: {r2_diff:.4f})",
    fontweight="bold",
)
plt.xlabel("Tarih")
plt.ylabel("USD Fiyatı")
plt.legend()
plt.grid(True, alpha=0.4)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================================
# 9a) PERFORMANS RAPORU
# ============================================================

print(f"\n PERFORMANS RAPORU (TÜM VERİ SETİ ÜZERİNDEN):")
print("=" * 50)
print(f"Eğitim MAE: {train_mae:.2f}")
print(f"Eğitim R²: {train_r2:.4f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Test R²: {test_r2:.4f}")

# Volatiliteyi hesaplama
predicted_daily_returns = np.diff(future_predictions) / future_predictions[:-1]
predicted_volatility = np.std(predicted_daily_returns) * 100

print(f"\n 30 GÜNLÜK TAHMİN ÖZETİ:")
print(f"Başlangıç Değeri: {last_actual:.2f}")
print(f"30 Gün Sonu Tahmin: {future_predictions[-1]:.2f}")
total_change_pct = (future_predictions[-1] - last_actual) / last_actual * 100
print(f"Toplam Değişim: {total_change_pct:+.2f}%")
print(
    f"Tahmin Volatilitesi: {predicted_volatility:.2f}% (Günlük % Değişim Standard Sapması)"
)

print(
    f"\n DÜZENLİLEŞTİRİLMİŞ MODEL ÇALIŞTIRMA VE GÖRSELLEŞTİRME SİSTEMİ BAŞARIYLA TAMAMLANDI."
)
