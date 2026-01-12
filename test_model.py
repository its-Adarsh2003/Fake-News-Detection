import pickle
import os
import sys

print("\n>>> test_model.py started")

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from preprocess import NewsPreprocessor
from features import FeatureExtractor

# load model + vectorizer
with open(os.path.join(BASE_DIR, "models", "best_model.pkl"), "rb") as f:
    model = pickle.load(f)
print(">>> model loaded")

pre = NewsPreprocessor()
ext = FeatureExtractor()
ext.load_tfidf(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))
print(">>> vectorizer loaded\n")

# Test cases
test_cases = [
    ("❌ FAKE EXAMPLE", "SHOCKING REVELATION!!! GOVERNMENT HIDING ANTI-AGING SERUM FROM PUBLIC!!! Big Pharma doesn't want you to know this ONE TRICK that will keep you young FOREVER!!! Celebrities use this secret discovered 20 years ago but THEY don't want you to know!!! SHARE THIS NOW before the government DELETES IT!!!"),
    ("✅ REAL EXAMPLE", "The Federal Reserve announced today that it will maintain interest rates at the current level following a two-day policy meeting. The decision comes as inflation shows signs of cooling while economic growth remains moderate."),
]

print("=" * 80)
for label, text in test_cases:
    print(f"\nTesting: {label}")
    clean = pre.clean_text(text)
    X = ext.extract_tfidf_features_transform([clean])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    result = "✅ REAL (1)" if pred == 1 else "❌ FAKE (0)"
    print(f"   Result: {result}")
    print(f"   Fake probability: {proba[0]:.4f}")
    print(f"   Real probability: {proba[1]:.4f}")
print("\n" + "=" * 80)
print(">>> done\n")
