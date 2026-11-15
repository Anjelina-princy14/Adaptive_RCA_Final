REMEDIATION_PLAYBOOK = {
    "Normal": "No action required. System operating within normal EMA latency baselines.",
    "Gateway_Timeout": "Scale API_Gateway worker threads, tune upstream timeout thresholds, and check load-balancer health.",
    "Authentication_Failure": "Verify Auth_Service token signing keys, Redis session cache connectivity, and OAuth provider latency.",
    "Database_Failure": "Increase Database connection pool size, terminate long-running locks, and inspect slow SQL queries.",
    "Core_Banking_Failure": "Inspect Core_Banking ledger microservice pods, check JVM/GC pauses, and verify message queue backlog.",
    "Network_Failure": "Check inter-service network switch packet loss, DNS resolution latency, and service mesh mTLS retries.",
    "External_Service_Failure": "Verify third-party payment switch / NPCI partner endpoint availability, enable circuit-breaker fallback, and review partner SLA timeouts."
}

def assess_severity_and_remediation(
    category: str,
    root_cause_service: str,
    total_latency_ms: float,
    recurrence_count: int,
    total_failed_txns: int
) -> dict:
    if category == "Normal":
        return {
            "severity_score": 0.0,
            "severity_level": "Low",
            "remediation": REMEDIATION_PLAYBOOK["Normal"]
        }

    latency_factor = min(1.0, total_latency_ms / 8000.0)
    recurrence_factor = min(1.0, recurrence_count / max(total_failed_txns, 1))
    criticality = 1.0 if root_cause_service in {"Core_Banking", "Database", "External_APIs"} else 0.75

    score = round(0.45 * latency_factor + 0.35 * criticality + 0.20 * recurrence_factor, 3)

    if score >= 0.75:
        level = "Critical"
    elif score >= 0.50:
        level = "High"
    elif score >= 0.25:
        level = "Medium"
    else:
        level = "Low"

    return {
        "severity_score": score,
        "severity_level": level,
        "remediation": REMEDIATION_PLAYBOOK.get(category, "Investigate correlated service logs.")
    }