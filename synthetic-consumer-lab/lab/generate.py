"""Calls a real multimodal model; never manufactures respondent answers."""
import base64
import mimetypes
import os
from openai import OpenAI

QUESTION = "Based on everything you've seen and heard, how likely are you to purchase this product?"


def generate_response(persona: dict, concept: dict, api_key: str, model: str = "gpt-4o-mini") -> str:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for live simulation")
    client = OpenAI(api_key=api_key, timeout=50.0, max_retries=2)
    system = ("You are participating in a consumer product concept survey. Imagine yourself as a consumer with "
              f"these characteristics: age {persona['age']}; gender {persona['gender']}; annual household income {persona['income']}; "
              f"US region {persona['region']}. Answer briefly in first person with realistic uncertainty. "
              "Do not produce a number, score, Likert rating, or percentage. You may dislike the concept. "
              "Base the response only on information presented; do not assert personal experience with it.")
    info = (f"Concept: {concept['name']}\nDescription: {concept['description']}\n"
            f"Benefits: {concept.get('benefits','')}\nPrice: {concept.get('price','not provided')}\n\n{QUESTION} "
            "Give one or two sentences describing your likely purchase behavior and the main reason.")
    parts = [{"type": "text", "text": info}]
    path = concept.get("image_path")
    if path and os.path.isfile(path):
        mime = mimetypes.guess_type(path)[0] or "image/png"
        if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            raise ValueError("Unsupported image type")
        with open(path, "rb") as f:
            image = base64.b64encode(f.read()).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}", "detail": "low"}})
    result = client.chat.completions.create(model=model, temperature=0.5, max_tokens=110,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": parts}])
    answer = (result.choices[0].message.content or "").strip()
    if not answer:
        raise RuntimeError("Empty LLM response")
    return answer
