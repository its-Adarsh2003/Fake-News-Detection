# bert_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class BertFakeNewsDetector:
    def __init__(self):
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text):
        inputs = self.tokenizer(
            text[:512],  # BERT max 512 tokens
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(logits, dim=1).item()
            confidence = probs[0][pred_label].item()
        
        return pred_label, confidence
