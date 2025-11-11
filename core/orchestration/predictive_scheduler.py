import time
from typing import Any, Dict


def predict_plan() -> Dict[str, Any]:
    hour = int(time.strftime("%H"))
    peak = 8 <= hour <= 11 or 14 <= hour <= 17
    allocate = {
        "analyst": 2 if peak else 1,
        "researcher": 2 if peak else 1,
        "conductor": 1,
        "coder": 1 if peak else 0,
    }
    preload = ["llama3.2:3b-instruct-q4_K_M"] + (
        ["mistral:7b-instruct-q4"] if peak else []
    )
    return {
        "allocate": allocate,
        "preload_models": preload,
        "notes": ["heuristics-v1", f"peak={peak}", f"hour={hour}"],
        "explain": {"qps_pred": 5 if peak else 1},
    }
