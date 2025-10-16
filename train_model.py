# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# --- 1. Load data ---
df = pd.read_csv("dummy_data_ukt_1000.csv")

# Optional: tampilkan ringkasan singkat
print("Jumlah baris:", len(df))
print(df.head())

# --- 2. Definisikan fitur & target ---
target_col = "Kategori_UKT"
X = df.drop(columns=[target_col, "ID_Mahasiswa"])
y = df[target_col]

# Encode target ke integer (UKT 1..6)
le = LabelEncoder()
y_enc = le.fit_transform(y)  # simpan le nanti bersama pipeline

# --- 3. Tentukan kolom numerik dan kategorikal ---
numeric_features = ["Pendapatan_Ortu", "Tanggungan_Keluarga", "Tagihan_Listrik", "Nilai_IPK"]
categorical_features = ["Pekerjaan_Ortu", "Kepemilikan_Rumah", "Kendaraan", "Beasiswa"]

# --- 4. Preprocessing pipeline ---
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="drop")

# --- 5. Model pipeline ---
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# --- 6. Train / test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# --- 7. Fit model ---
print("Melatih model RandomForest...")
clf.fit(X_train, y_train)

# --- 8. Evaluasi ---
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy (test): {acc:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# --- 9. Simpan pipeline + label encoder ---
model_filename = "model_pipeline.joblib"
joblib.dump({"pipeline": clf, "label_encoder": le}, model_filename)
print(f"Model pipeline disimpan ke: {model_filename}")

# --- 10. (Opsional) Tampilkan pentingnya fitur (approx) ---
# Untuk RandomForest kita bisa ambil fitur setelah OHE; bangun nama fitur:
ohe = clf.named_steps["preprocessor"].named_transformers_["cat"].named_steps["ohe"]
ohe_feature_names = []
for i, col in enumerate(categorical_features):
    # get categories for col
    cats = ohe.categories_[i]
    ohe_feature_names += [f"{col}__{c}" for c in cats]

feature_names = numeric_features + ohe_feature_names
importances = clf.named_steps["classifier"].feature_importances_
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print("\nTop 10 fitur penting:")
for name, imp in feat_imp[:10]:
    print(f"{name}: {imp:.4f}")
