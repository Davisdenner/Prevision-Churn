import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ModelMonitor:
    window_size: int = 1000
    _predictions: deque = field(default_factory=lambda: deque(maxlen=1000))
    _latencies_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    _total_requests: int = 0
    _total_errors: int = 0
    _start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def record_prediction(self, probability: float, latency_ms: float):
        self._predictions.append(probability)
        self._latencies_ms.append(latency_ms)
        self._total_requests += 1

    def record_error(self):
        self._total_errors += 1
        self._total_requests += 1

    def get_metrics(self) -> dict:
        preds = list(self._predictions)
        latencies = list(self._latencies_ms)

        if not preds:
            return {
                "status": "no_predictions_yet",
                "total_requests": self._total_requests,
                "uptime_since": self._start_time.isoformat(),
            }

        preds_arr = np.array(preds)
        latencies_arr = np.array(latencies)

        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": round(self._total_errors / max(self._total_requests, 1), 4),
            "uptime_since": self._start_time.isoformat(),
            "prediction_stats": {
                "window_size": len(preds),
                "mean_probability": round(float(preds_arr.mean()), 4),
                "std_probability": round(float(preds_arr.std()), 4),
                "median_probability": round(float(np.median(preds_arr)), 4),
                "high_risk_rate": round(float((preds_arr >= 0.7).mean()), 4),
                "medium_risk_rate": round(float(((preds_arr >= 0.4) & (preds_arr < 0.7)).mean()), 4),
                "low_risk_rate": round(float((preds_arr < 0.4).mean()), 4),
            },
            "latency_ms": {
                "mean": round(float(latencies_arr.mean()), 2),
                "p50": round(float(np.percentile(latencies_arr, 50)), 2),
                "p95": round(float(np.percentile(latencies_arr, 95)), 2),
                "p99": round(float(np.percentile(latencies_arr, 99)), 2),
                "max": round(float(latencies_arr.max()), 2),
            },
        }


monitor = ModelMonitor()