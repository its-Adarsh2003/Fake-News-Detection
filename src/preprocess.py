# ADD THIS TO YOUR preprocess.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class NewsPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.contractions_dict = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "i'm": "i am", "isn't": "is not", "doesn't": "does not",
            "hasn't": "has not", "haven't": "have not", "wasn't": "was not"
        }
    
    def expand_contractions(self, text):
        """Expand contractions for better preprocessing"""
        for key, value in self.contractions_dict.items():
            text = text.replace(key, value)
        return text
    
    def clean_text(self, text):
        """Clean and preprocess news text"""
        if not isinstance(text, str):
            text = str(text)
        
        # 1. Expand contractions
        text = self.expand_contractions(text.lower())
        
        # 2. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 3. Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # 4. Remove special chars (keep only letters + spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 5. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 6. Tokenize + remove stopwords + stemming
        tokens = text.split()
        tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
