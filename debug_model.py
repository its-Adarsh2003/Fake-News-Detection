# -*- coding: utf-8 -*-
import numpy as np
import pickle
from src.preprocess import NewsPreprocessor
from src.features import FeatureExtractor

print("MODEL DEBUG TEST")
print("=" * 50)

# Load model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)
print("OK Model loaded")

extractor = FeatureExtractor()
extractor.load_tfidf("models/tfidf_vectorizer.pkl")
print("OK Vectorizer loaded")

preprocessor = NewsPreprocessor()
print("OK Preprocessor loaded")

print("\nTESTING SAMPLES:")
print("-" * 50)

# Real News
real_sample = "The Federal Reserve announced today that it will maintain interest rates."
real_clean = preprocessor.clean_text(real_sample)
X_real = extractor.extract_tfidf_features_transform([real_clean])
pred_real = model.predict(X_real)[0]

print("\nREAL NEWS:")
print("Prediction:", pred_real, "(0=Fake, 1=Real)")
if hasattr(model, 'predict_proba'):
    prob = model.predict_proba(X_real)[0]
    print("Confidence:", prob[pred_real])

# Fake News
fake_sample = "SHOCKING!!! GOVERNMENT HIDING SECRET!!! SHARE NOW!!!"
fake_clean = preprocessor.clean_text(fake_sample)
X_fake = extractor.extract_tfidf_features_transform([fake_clean])
pred_fake = model.predict(X_fake)[0]

print("\nFAKE NEWS:")
print("Prediction:", pred_fake, "(0=Fake, 1=Real)")
if hasattr(model, 'predict_proba'):
    prob = model.predict_proba(X_fake)[0]
    print("Confidence:", prob[pred_fake])

print("\n" + "=" * 50)
if pred_real == 1 and pred_fake == 0:
    print("OK - MODEL WORKING CORRECTLY")
elif pred_real == 0 and pred_fake == 1:
    print("ERROR - MODEL IS INVERTED")
else:
    print("UNCLEAR - Check training")
