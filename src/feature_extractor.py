import pandas as pd

def extract_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tx_id, group in df.groupby("transaction_id", sort=False):
        codes = set(group["error_code"])
        svc_lat = group.groupby("service_name")["latency_ms"].max().to_dict()

        rows.append({
            "transaction_id": tx_id,
            "total_latency_ms": group["latency_ms"].sum(),
            "max_latency_ms": group["latency_ms"].max(),
            "mean_norm_latency": group["norm_latency"].mean(),
            "total_retries": group["retry_count"].sum(),
            "failed_hop_count": group["is_error"].sum(),
            "has_504_gateway": int("504_GATEWAY_TIMEOUT" in codes),
            "has_401_auth": int("401_AUTH_INVALID" in codes),
            "has_502_core_bank": int("502_CORE_BANK_ERR" in codes),
            "has_503_db": int("503_DB_CONN_POOL_EXHAUSTED" in codes),
            "has_599_network": int("599_NET_PACKET_LOSS" in codes),
            "has_502_external": int("502_EXT_PARTNER_UNAVAILABLE" in codes),
            "gateway_latency": svc_lat.get("API_Gateway", 0.0),
            "auth_latency": svc_lat.get("Auth_Service", 0.0),
            "payment_latency": svc_lat.get("Payment_Service", 0.0),
            "db_latency": svc_lat.get("Database", 0.0),
            "core_bank_latency": svc_lat.get("Core_Banking", 0.0),
            "external_latency": svc_lat.get("External_APIs", 0.0),
            "ground_truth_category": group["ground_truth_category"].iloc[0],
            "ground_truth_root_cause": group["ground_truth_root_cause"].iloc[0],
        })
    return pd.DataFrame(rows)