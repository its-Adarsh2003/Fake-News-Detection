import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class FeatureExtractor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            max_df=0.80,
            min_df=2,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english'
        )
        
    def extract_tfidf_features(self, texts):
        tfidf_matrix = self.tfidf.fit_transform(texts)
        return tfidf_matrix
    
    def extract_tfidf_features_transform(self, texts):
        return self.tfidf.transform(texts)
    
    def save_tfidf(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.tfidf, f)
    
    def load_tfidf(self, path):
        with open(path, 'rb') as f:
            self.tfidf = pickle.load(f)
