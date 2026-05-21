import pandas as pd
from sklearn.metrics import precision_score, recall_score

def calculate_accuracy_metrics(test_cases):
    if not test_cases:
        return None
    
    df = pd.DataFrame(test_cases)
    # Define success as an exact match
    df['vendor_match'] = (df['vendor_exp'] == df['vendor_ext']).astype(int)
    df['total_match'] = (df['total_exp'] == df['total_ext']).astype(int)
    
    # Ground truth: we expect all extractions to be correct (1)
    y_true = [1] * len(df)
    
    metrics = {
        "vendor_precision": precision_score(y_true, df['vendor_match'], zero_division=0),
        "vendor_recall": recall_score(y_true, df['vendor_match'], zero_division=0),
        "total_precision": precision_score(y_true, df['total_match'], zero_division=0),
        "count": len(df)
    }
    return metrics, df