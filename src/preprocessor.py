import json
import pandas as pd

def load_and_preprocess_logs(json_path: str):
    with open(json_path, "r") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df["is_error"] = (df["status"] == "FAILED").astype(int)

    df["norm_latency"] = df.groupby("service_name")["latency_ms"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    )

    df = df.sort_values(by=["transaction_id", "hop_order"]).reset_index(drop=True)
    traces = {tx_id: group.to_dict(orient="records") for tx_id, group in df.groupby("transaction_id")}
    return df, traces