"""Thin adapter to the published PyMC Labs ResponseRater.

IMPORTANT: The paper says six anchor sets were manually optimized, but does not
publish their literal text. These are explicitly PROTOTYPE anchors, not the
paper's original sets. Upstream's default sentence-transformer encoder may
also differ from the paper's OpenAI text-embedding-3-small encoder.
"""
import numpy as np
from .core import validate_pmf

ANCHORS = [
 ["I definitely would not buy this product.", "I probably would not buy this product.", "I am undecided whether I would buy this product.", "I probably would buy this product.", "I definitely would buy this product."],
 ["I have no interest in purchasing this.", "I have little interest in purchasing this.", "I could go either way about purchasing this.", "I am interested in purchasing this.", "I am extremely interested in purchasing this."],
 ["I would avoid buying this.", "I doubt I would try buying this.", "I might or might not try buying this.", "I would consider buying this.", "I would certainly buy this."],
 ["There is almost no chance I will purchase this.", "There is a small chance I will purchase this.", "There is an even chance I will purchase this.", "There is a good chance I will purchase this.", "There is an extremely high chance I will purchase this."],
 ["This is not something I would spend money on.", "I am reluctant to spend money on this.", "I am unsure about spending money on this.", "I would be willing to spend money on this.", "I would happily spend money on this."],
 ["I cannot imagine choosing this product.", "I would be unlikely to choose this product.", "I am neutral about choosing this product.", "I would be likely to choose this product.", "I would confidently choose this product."],
]


def references():
    import polars as pl
    return pl.DataFrame([{"id": f"anchor_{i+1}", "int_response": score, "sentence": sentence}
                         for i, group in enumerate(ANCHORS) for score, sentence in enumerate(group, 1)])


def score_responses(responses: list[str]) -> list[list[float]]:
    if not responses or any(not isinstance(r, str) or not r.strip() for r in responses):
        raise ValueError("Responses must be nonempty text")
    from semantic_similarity_rating import ResponseRater
    rater = ResponseRater(references())
    values = rater.get_response_pmfs(reference_set_id="mean", llm_responses=responses, temperature=1.0, epsilon=0.0)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(responses), 5):
        raise ValueError(f"Unexpected SSR shape {values.shape}")
    return [validate_pmf(p) for p in values]
