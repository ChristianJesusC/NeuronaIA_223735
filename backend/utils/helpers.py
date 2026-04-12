import math


def safe_float(v, fallback: float = 0.0) -> float:
    try:
        f = float(v)
        return fallback if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return fallback


def safe_list(lst) -> list:
    return [safe_float(v) for v in lst]
