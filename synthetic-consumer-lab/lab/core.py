"""Panel construction, transparent aggregation and export routines."""
from __future__ import annotations
import json
import random
from collections import Counter
from typing import Any
import numpy as np

REGIONS = ["Northeast", "Midwest", "South", "West"]
GENDERS = ["Woman", "Man", "Nonbinary"]
INCOMES = ["Under $35k", "$35k–$75k", "$75k–$150k", "Over $150k"]


def weighted_choice(rng: random.Random, items: list[str], weights: list[float]) -> str:
    if len(items) != len(weights) or not items or any(x < 0 for x in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be nonnegative, match categories, and sum above zero")
    return rng.choices(items, weights=weights, k=1)[0]


def create_panel(n: int, age_min: int, age_max: int, weights: dict[str, list[float]] | None = None, seed: int = 42) -> list[dict[str, Any]]:
    if not 1 <= n <= 1000 or age_min < 18 or age_max > 100 or age_min > age_max:
        raise ValueError("Invalid panel parameters")
    weights = weights or {}
    rng = random.Random(seed)
    result = []
    for i in range(n):
        result.append({"respondent_id": f"P{i+1:04d}", "age": rng.randint(age_min, age_max),
            "gender": weighted_choice(rng, GENDERS, weights.get("gender", [49, 49, 2])),
            "income": weighted_choice(rng, INCOMES, weights.get("income", [24, 36, 28, 12])),
            "region": weighted_choice(rng, REGIONS, weights.get("region", [18, 23, 38, 21]))})
    return result


def validate_pmf(pmf: Any) -> list[float]:
    a = np.asarray(pmf, dtype=float)
    if a.shape != (5,) or not np.isfinite(a).all() or np.any(a < -1e-10) or abs(float(a.sum()) - 1.0) > 1e-5:
        raise ValueError(f"SSR returned invalid distribution: {pmf}")
    return [float(max(0, x)) for x in a]


def aggregate(records: list[dict], filters: dict | None = None) -> list[dict]:
    filters = filters or {}
    selected = [r for r in records if r.get("pmf") and all(r.get("persona", {}).get(k) == v for k, v in filters.items() if v != "All")]
    grouped: dict[str, list] = {}
    for record in selected:
        grouped.setdefault(record["concept_id"], []).append(validate_pmf(record["pmf"]))
    results = []
    for concept_id, pmfs in grouped.items():
        p = np.asarray(pmfs).mean(axis=0)
        results.append({"concept_id": concept_id, "count": len(pmfs), "mean": float(np.dot(p, np.arange(1, 6))),
                        "top_two": float(p[3] + p[4]), "bottom_two": float(p[0] + p[1]), "distribution": p.tolist()})
    return sorted(results, key=lambda x: x["mean"], reverse=True)


def record_csv_rows(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        p = r.get("pmf") or [None] * 5
        row = {"concept_id": r["concept_id"], "respondent_id": r["respondent_id"], "response": r.get("response", ""), "status": r.get("status", ""), "error": r.get("error", "")}
        row.update(r.get("persona", {}))
        row.update({f"rating_{i}_probability": p[i-1] for i in range(1, 6)})
        rows.append(row)
    return rows


def diagnostic_words(records: list[dict], max_words: int = 18) -> list[tuple[str, int]]:
    """Transparent term frequencies, not inferred consumer motivations."""
    stop = set("that this with from would could should about after before just very maybe might really likely purchase product like think have there they them will some what when where only more less want need into than because also it's i'm it's the and for you but are was were not too its one two can get buy buying".split())
    import re
    counts = Counter(word.lower() for r in records for word in re.findall(r"[A-Za-z]{4,}", r.get("response", "")) if word.lower() not in stop)
    return counts.most_common(max_words)
