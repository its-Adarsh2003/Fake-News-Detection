import numpy as np
import pandas as pd
import pickle
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# src import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import NewsPreprocessor
from features import FeatureExtractor

print("=" * 80)
print("🚀 FAKE NEWS DETECTION - TRAINING PIPELINE (RF + LR + SVM)")
print("=" * 80)

# ---- STEP 1: LOAD DATA ----
print("\n📥 Step 1: Loading data...")

fake_path = "data\\News_dataset\\Fake.csv"
true_path = "data\\News_dataset\\True.csv"

try:
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    print(f"✅ Loaded {len(fake)} fake and {len(true)} real articles")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# 0 = fake, 1 = real
fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], ignore_index=True).reset_index(drop=True)
print(f"✅ Combined dataset size: {len(df)}")
print("Label distribution:", df["label"].value_counts().to_dict())

# ---- STEP 2: CLEAN ----
print("\n🔍 Step 2: Cleaning data...")
df = df.dropna(subset=["text", "label"])
print(f"✅ After null removal: {len(df)} articles")

# ---- STEP 3: PREPROCESS ----
print("\n🧹 Step 3: Preprocessing text...")
pre = NewsPreprocessor()
df["clean_text"] = df["text"].apply(lambda x: pre.clean_text(str(x)))
print("✅ Text preprocessing complete")

# ---- STEP 4: FEATURES ----
print("\n📊 Step 4: Extracting TF-IDF features...")
extractor = FeatureExtractor()
X = extractor.extract_tfidf_features(df["clean_text"])
y = df["label"].values
print("Label distribution:", dict(zip(*np.unique(y, return_counts=True))))
print(f"✅ Feature extraction complete - Shape: {X.shape}")

# ---- STEP 5: SPLIT ----
print("\n✂️  Step 5: Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ---- STEP 6: DEFINE MODELS ----
models = {}

# 1) Random Forest
models["Random_Forest"] = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

# 2) Logistic Regression
models["Logistic_Regression"] = LogisticRegression(
    max_iter=300,
    class_weight="balanced",
    n_jobs=-1,
)

# 3) Linear SVM (calibrated for probabilities)
svm_base = LinearSVC(class_weight="balanced", random_state=42)
models["Linear_SVM"] = CalibratedClassifierCV(svm_base, cv=3)

# ---- STEP 7: TRAIN + EVAL ALL ----
results = {}
best_name = None
best_f1 = -1
best_model = None

for name, clf in models.items():
    print(f"\n🤖 Training model: {name}")
    clf.fit(X_train, y_train)
    print(f"✅ {name} training complete")

    y_pred = clf.predict(X_test)

    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = float("nan")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"   Accuracy : {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall   : {recall*100:.2f}%")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   ROC-AUC  : {roc_auc:.4f}")

    results[name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }

    if f1 > best_f1:
        best_f1 = f1
        best_name = name
        best_model = clf

print("\nBest model based on F1:", best_name, f"({best_f1:.4f})")

# ---- STEP 8: SAVE BEST MODEL + VECTORIZER ----
print("\n💾 Step 8: Saving best model & TF-IDF...")
os.makedirs("models", exist_ok=True)

with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print(f"✅ Best model saved as: {best_name}")

extractor.save_tfidf("models/tfidf_vectorizer.pkl")
print("✅ TF-IDF vectorizer saved")

# ---- STEP 9: SAVE RESULTS FOR DASHBOARD ----
print("\n📊 Step 9: Saving all model results...")
os.makedirs("results", exist_ok=True)

import json
with open("results/model_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("✅ Results saved to: results/model_results.json")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE! Ready for Streamlit app!")
print("=" * 80)
