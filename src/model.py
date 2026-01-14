from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import numpy as np
import pandas as pd

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # ROC-AUC (if model supports probabilities)
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = np.nan
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc)
    }
    
    print(f"\n📊 {model_name} Results:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    
    return metrics

def print_confusion_matrix(y_test, y_pred):
    """Pretty print confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    print("\n📋 Confusion Matrix:")
    print("           Predicted")
    print("Actual   |  Fake  | Real")
    print("---------|--------|------")
    print(f"Fake     | {cm[0,0]:6d} | {cm[0,1]:5d}")
    print(f"Real     | {cm[1,0]:6d} | {cm[1,1]:5d}")

def get_feature_importance(model, feature_names, top_k=10):
    """Get top K important features (for tree-based models)"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_k:][::-1]
        print(f"\n⭐ Top {top_k} Features:")
        for i, idx in enumerate(indices, 1):
            print(f"   {i}. {feature_names[idx]}: {importances[idx]:.4f}")
