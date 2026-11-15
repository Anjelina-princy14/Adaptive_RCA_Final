import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.preprocessor import load_and_preprocess_logs
from src.feature_extractor import extract_transaction_features
from src.rca_engine import AdaptiveRCAEngine

def run_evaluation():
    # 1. Load & Preprocess
    df, traces = load_and_preprocess_logs("data/synthetic_transactions.json")
    features_df = extract_transaction_features(df)

    # 2. Train & Classify Failure Categories
    engine = AdaptiveRCAEngine(alpha=0.10)
    X_test, y_test = engine.train_classifier(features_df)
    y_pred = engine.classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100

    print("=== FAILURE CLASSIFICATION METRICS (Test Set) ===")
    print(f"Accuracy  : {acc:.2f}%")
    print(f"Precision : {prec:.2f}%")
    print(f"Recall    : {rec:.2f}%")
    print(f"F1-Score  : {f1:.2f}%\n")

    # 3. Evaluate Root Cause Identification Accuracy across all failed transactions (Eq. 6)
    failed_correct = 0
    total_failed = 0

    for tx_id, events in traces.items():
        gt_rc = events[0]["ground_truth_root_cause"]
        res = engine.identify_root_cause(events)
        pred_rc = res["root_cause_service"]

        if gt_rc != "None":
            total_failed += 1
            if pred_rc == gt_rc:
                failed_correct += 1

    rca_accuracy = (failed_correct / total_failed) * 100 if total_failed > 0 else 0.0
    print("=== ROOT CAUSE IDENTIFICATION METRICS (All 600 Failed Txns) ===")
    print(f"Correct Root Causes : {failed_correct} / {total_failed}")
    print(f"RCA Accuracy        : {rca_accuracy:.2f}%")

if __name__ == "__main__":
    run_evaluation()