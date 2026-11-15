import os
import json
import random
from datetime import datetime, timedelta

# 7 unique services (one event per service per transaction)
SERVICE_NODES = [
    ("Client_App", ["API_Gateway"], 25.0),
    ("API_Gateway", ["Auth_Service"], 35.0),
    ("Auth_Service", ["Payment_Service"], 60.0),
    ("Payment_Service", ["Database", "Core_Banking"], 45.0),
    ("Database", [], 30.0),
    ("Core_Banking", ["External_APIs"], 120.0),
    ("External_APIs", [], 150.0),
]

SCENARIO_CONFIG = {
    "Normal": ("None", "200_OK", "Processed normally"),
    "Gateway_Timeout": ("API_Gateway", "504_GATEWAY_TIMEOUT", "API_Gateway timed out waiting on upstream response"),
    "Authentication_Failure": ("Auth_Service", "401_AUTH_INVALID", "Token validation failed in Auth_Service"),
    "Database_Failure": ("Database", "503_DB_CONN_POOL_EXHAUSTED", "Database connection pool exhausted"),
    "Core_Banking_Failure": ("Core_Banking", "502_CORE_BANK_ERR", "Core_Banking ledger service returned 502"),
    "Network_Failure": ("Core_Banking", "599_NET_PACKET_LOSS", "Severe packet loss between Payment_Service and Core_Banking"),
    "External_Service_Failure": ("External_APIs", "502_EXT_PARTNER_UNAVAILABLE", "Third-party NPCI/partner endpoint unavailable"),
}

def generate_transaction_events(tx_id: str, base_time: datetime, scenario: str) -> list:
    gt_root, fault_code, fault_msg = SCENARIO_CONFIG[scenario]

    # Propagate failures upstream along caller -> dependency graph
    failed_services = set()
    if gt_root != "None":
        failed_services.add(gt_root)
        changed = True
        while changed:
            changed = False
            for svc, deps, _ in SERVICE_NODES:
                if any(d in failed_services for d in deps) and svc not in failed_services:
                    failed_services.add(svc)
                    changed = True

    events = []
    for hop_idx, (svc, deps, base_lat) in enumerate(SERVICE_NODES, start=1):
        ts = (base_time + timedelta(milliseconds=hop_idx * 15)).isoformat()
        dep_str = ",".join(deps) if deps else "None"

        if svc == gt_root:
            status = "FAILED"
            error_code = fault_code
            if scenario == "Authentication_Failure":
                latency = round(base_lat * random.uniform(0.9, 1.35), 2)
            else:
                latency = round(base_lat * random.uniform(14.0, 24.0), 2)
            retries = random.randint(2, 3)
            msg = fault_msg
        elif svc in failed_services:
            failed_deps = [d for d in deps if d in failed_services]
            status = "FAILED"
            error_code = "504_EXT_GATEWAY_TIMEOUT"
            latency = round(base_lat * random.uniform(12.0, 28.0), 2)
            retries = random.randint(0, 2)
            msg = f"{svc} timed out waiting on downstream dependency ({','.join(failed_deps)})"
        else:
            status = "SUCCESS"
            error_code = "200_OK"
            latency = round(base_lat * random.uniform(0.85, 1.25), 2)
            retries = 0
            msg = f"{svc} processed request normally"

        events.append({
            "transaction_id": tx_id,
            "timestamp": ts,
            "hop_order": hop_idx,
            "service_name": svc,
            "dependency": dep_str,
            "latency_ms": latency,
            "status": status,
            "error_code": error_code,
            "retry_count": retries,
            "log_message": msg,
            "ground_truth_category": scenario,
            "ground_truth_root_cause": gt_root
        })
    return events

def generate_dataset(output_path="data/synthetic_transactions.json"):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    distribution = [
        ("Normal", 300),
        ("Gateway_Timeout", 150),
        ("Core_Banking_Failure", 120),
        ("Authentication_Failure", 100),
        ("Database_Failure", 90),
        ("Network_Failure", 80),
        ("External_Service_Failure", 60),
    ]
    all_records = []
    tx_counter = 1
    start_time = datetime(2026, 11, 1, 10, 0, 0)

    for scenario, count in distribution:
        for _ in range(count):
            tx_id = f"TXN_{tx_counter:04d}"
            events = generate_transaction_events(
                tx_id, start_time + timedelta(seconds=tx_counter * 5), scenario
            )
            all_records.extend(events)
            tx_counter += 1

    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Generated {tx_counter - 1} transactions ({len(all_records)} events) at {output_path}")

if __name__ == "__main__":
    generate_dataset()