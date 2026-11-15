from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ALPHA = 0.10
W_ERROR = 0.35
W_LATENCY = 0.25
W_DEPENDENCY = 0.25
W_TEMPORAL = 0.15

ALL_FEATURE_COLS = [
    "total_latency_ms", "max_latency_ms", "mean_norm_latency",
    "total_retries", "failed_hop_count",
    "has_504_gateway", "has_401_auth", "has_502_core_bank",
    "has_503_db", "has_599_network", "has_502_external",
    "gateway_latency", "auth_latency", "payment_latency",
    "db_latency", "core_bank_latency", "external_latency"
]

BEHAVIORAL_FEATURE_COLS = [
    "total_latency_ms", "max_latency_ms", "mean_norm_latency",
    "total_retries", "failed_hop_count",
    "gateway_latency", "auth_latency", "payment_latency",
    "db_latency", "core_bank_latency", "external_latency"
]

DIRECT_FAULT_CODES = {
    "504_GATEWAY_TIMEOUT",
    "401_AUTH_INVALID",
    "502_CORE_BANK_ERR",
    "503_DB_CONN_POOL_EXHAUSTED",
    "599_NET_PACKET_LOSS",
    "502_EXT_PARTNER_UNAVAILABLE"
}

DEFAULT_EMA_BASELINES = {
    "Client_App": 25.0,
    "API_Gateway": 35.0,
    "Payment_Service": 45.0,
    "Auth_Service": 60.0,
    "Core_Banking": 120.0,
    "Database": 30.0,
    "External_APIs": 150.0
}

class AdaptiveRCAEngine:
    def __init__(self, alpha=ALPHA):
        self.alpha = alpha
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ema_baselines = dict(DEFAULT_EMA_BASELINES)

    def load_ema_baselines(self, saved_baselines: dict):
        for svc, val in saved_baselines.items():
            if svc in self.ema_baselines:
                self.ema_baselines[svc] = float(val)

    def get_ema_baselines(self) -> dict:
        return dict(self.ema_baselines)

    def train_classifier(self, features_df: pd.DataFrame, feature_cols=ALL_FEATURE_COLS):
        X = features_df[feature_cols]
        y = features_df["ground_truth_category"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        self.classifier.fit(X_train, y_train)
        return X_test, y_test

    def predict_failure_category(self, features_df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.classifier.predict(features_df[ALL_FEATURE_COLS]), index=features_df.index)

    def apply_post_diagnosis_ema_update(self, tx_events: list):
        for e in tx_events:
            if e["status"] == "SUCCESS":
                svc = e["service_name"]
                prev_b = self.ema_baselines[svc]
                self.ema_baselines[svc] = (1 - self.alpha) * prev_b + self.alpha * e["latency_ms"]

    def identify_root_cause(self, tx_events: list, update_ema: bool = False) -> dict:
        failed_events = [e for e in tx_events if e["status"] == "FAILED"]
        if not failed_events:
            result = {
                "root_cause_service": "None",
                "root_cause_score": 0.0,
                "confidence": 100.0,
                "is_uncertain": False,
                "explanation": "All 7 microservices responded normally within their adaptive EMA latency baselines.",
                "evidence": {}
            }
            if update_ema:
                self.apply_post_diagnosis_ema_update(tx_events)
            return result

        failed_services = {e["service_name"] for e in failed_events}
        service_all_deps = {}
        for e in tx_events:
            raw_dep = str(e.get("dependency", "None"))
            deps = {d.strip() for d in raw_dep.split(",") if d.strip() and d.strip() != "None"}
            service_all_deps.setdefault(e["service_name"], set()).update(deps)

        failed_map = {e["service_name"]: e for e in failed_events}

        def parse_ts(val) -> float:
            if isinstance(val, (int, float)):
                return float(val)
            return datetime.fromisoformat(str(val)).timestamp()

        leaf_timestamps = [
            parse_ts(e["timestamp"])
            for s, e in failed_map.items()
            if not (service_all_deps.get(s, set()) & failed_services)
        ]
        all_timestamps = [parse_ts(e["timestamp"]) for e in failed_map.values()]
        earliest_leaf_ts = min(leaf_timestamps) if leaf_timestamps else min(all_timestamps)
        ts_span = max(max(all_timestamps) - min(all_timestamps), 0.001)

        candidate_scores = {}
        evidence_breakdown = {}

        for s, e in failed_map.items():
            code = e["error_code"]
            dep_failed = bool(service_all_deps.get(s, set()) & failed_services)

            if code in DIRECT_FAULT_CODES and not dep_failed:
                e_score = 1.0
            elif code == "504_EXT_GATEWAY_TIMEOUT" and dep_failed:
                e_score = 0.20
            else:
                e_score = 0.45

            baseline = self.ema_baselines[s]
            obs_lat = e["latency_ms"]
            ratio = obs_lat / max(baseline, 1.0)
            l_score = min(1.0, max(0.0, (ratio - 1.2) / 10.0))

            d_score = 0.0 if dep_failed else 1.0

            if dep_failed:
                t_score = 0.15
            else:
                ev_ts = parse_ts(e["timestamp"])
                t_score = max(0.2, 1.0 - (abs(ev_ts - earliest_leaf_ts) / ts_span))

            r_score = (
                W_ERROR * e_score +
                W_LATENCY * l_score +
                W_DEPENDENCY * d_score +
                W_TEMPORAL * t_score
            )
            candidate_scores[s] = round(r_score, 4)
            evidence_breakdown[s] = {
                "E(s)": round(e_score, 2),
                "L(s)": round(l_score, 2),
                "D(s)": round(d_score, 2),
                "T(s)": round(t_score, 2),
                "R(s)": round(r_score, 4),
                "EMA_Baseline_ms": round(baseline, 2),
                "Observed_Latency_ms": obs_lat
            }

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        best_service, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score

        confidence = round(min(99.9, max(50.0, (best_score * 70.0) + (margin * 50.0))), 1)
        is_uncertain = (margin < 0.12 and len(ranked) > 1) or best_score < 0.55

        best_ev = evidence_breakdown[best_service]
        obs_lat = best_ev["Observed_Latency_ms"]
        base_lat = best_ev["EMA_Baseline_ms"]

        if obs_lat > base_lat * 1.2:
            latency_phrase = f"Its observed latency ({obs_lat} ms) exceeded its EMA baseline ({base_lat} ms)"
        else:
            latency_phrase = f"Its observed latency ({obs_lat} ms) did not exceed the configured EMA threshold ({base_lat * 1.2:.2f} ms)"

        other_failed = [s for s in candidate_scores if s != best_service]
        cascade_note = (
            f" Other affected services: {', '.join(other_failed)}."
            if other_failed else " No other services failed in this trace."
        )
        uncertain_prefix = "[UNCERTAIN — Competing Evidence] " if is_uncertain else ""

        explanation = (
            f"{uncertain_prefix}{best_service} achieved the highest root-cause score R(s) = {best_score} "
            f"(Confidence: {confidence}%). {latency_phrase} with error code "
            f"'{failed_map[best_service]['error_code']}' and leaf dependency score D(s) = {best_ev['D(s)']}.{cascade_note}"
        )

        result = {
            "root_cause_service": best_service,
            "root_cause_score": best_score,
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "explanation": explanation,
            "evidence": evidence_breakdown
        }

        if update_ema:
            self.apply_post_diagnosis_ema_update(tx_events)

        return result