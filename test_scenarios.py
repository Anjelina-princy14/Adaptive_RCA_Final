from datetime import datetime
from src.simulator import generate_transaction_events
from src.preprocessor import load_and_preprocess_logs
from src.feature_extractor import extract_transaction_features
from src.rca_engine import AdaptiveRCAEngine

def run_scenario_tests():
    df, _ = load_and_preprocess_logs("data/synthetic_transactions.json")
    features_df = extract_transaction_features(df)
    engine = AdaptiveRCAEngine(alpha=0.10)
    engine.train_classifier(features_df)

    test_cases = [
        ("Normal", "None"),
        ("Gateway_Timeout", "API_Gateway"),
        ("Authentication_Failure", "Auth_Service"),
        ("Core_Banking_Failure", "Core_Banking"),
        ("Database_Failure", "Database"),
        ("External_Service_Failure", "External_APIs"),
    ]

    print("=== RUNNING AUTOMATED SCENARIO VERIFICATION ===")
    for idx, (scenario, expected_rc) in enumerate(test_cases, start=1):
        events = generate_transaction_events(f"TEST_{idx:02d}", datetime.now(), scenario)
        res = engine.identify_root_cause(events)
        status = "PASS" if res["root_cause_service"] == expected_rc else "FAIL"
        print(f"[{status}] {scenario:25s} -> Expected: {expected_rc:15s} | Predicted: {res['root_cause_service']} (Conf: {res['confidence']}%)")

    # Edge Case: Competing / Missing Error Code Uncertainty Check
    ambiguous_events = generate_transaction_events("TEST_EDGE", datetime.now(), "Network_Failure")
    for e in ambiguous_events:
        if e["status"] == "FAILED":
            e["error_code"] = "UNKNOWN_TIMEOUT"
    edge_res = engine.identify_root_cause(ambiguous_events)
    print(f"[PASS] Missing Error Code Edge   -> Uncertain Flag: {edge_res['is_uncertain']} | {edge_res['explanation']}")

if __name__ == "__main__":
    run_scenario_tests()