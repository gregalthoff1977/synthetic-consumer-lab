# Synthetic Consumer Concept Testing Lab

A runnable Streamlit prototype for generating **real API-backed synthetic consumer responses** and converting them into probabilistic purchase-intent ratings with the [PyMC Labs Semantic Similarity Rating (SSR) package](https://github.com/pymc-labs/semantic-similarity-rating).

**This is exploratory synthetic research, not a validated consumer-demand or sales forecast.**

## What works

- Create and save studies with 2–5 concepts, descriptions, prices and optional images
- Generate fixed, reproducible demographic personas (same panel for all concepts)
- Call a live vision-capable OpenAI model for first-person free-text responses
- Resume partially completed runs, save each response in SQLite
- Score real responses through PyMC Labs' `ResponseRater` using six local anchor sets
- Compare expected ratings, top-two box, and all five probability categories
- Slice by persona geography, gender and income; inspect original responses
- Download respondent CSV, summary CSV, complete JSON
- Automated tests for generation-independent components

## Important deviations from the published study

1. The PDF describes six manually tuned sets of reference statements but **does not publish their exact wording**. This repository uses six transparent, manually authored *prototype* sets. They have not been calibrated against human research.
2. The upstream package's default sentence-transformer embedding encoder is used, **not** the paper's `text-embedding-3-small` model. This limits any claim of replication. The package is not replaced with a custom scoring algorithm.
3. The source study evaluated 57 personal-care concepts; transfer to beverages is unvalidated.
4. Persona weights are illustrative, not census-representative.
5. Segment results, especially small groups, should not be interpreted as market-level estimates.
6. A fixed persona and generation parameters do not make LLM responses fully reproducible.
7. CSV term-frequency diagnostics are **not** AI-derived market insights or human quotes.

## Run locally

Python 3.11 recommended. Git must be installed to install the upstream scoring package.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
streamlit run app.py
```

The first SSR scoring run may download sentence-transformer model weights; this requires internet access and may take time. GPU is not required for the intended small-scale use, but CPU inference may be slow.

Alternatively provide the API key via Streamlit's password input. Never commit real keys. The OpenAI API is separately billed; a ChatGPT subscription does not include API usage.

## Deploy

1. Push this repository to GitHub.
2. Connect the repository to Streamlit Community Cloud or a compatible host.
3. Set app entrypoint to `app.py` and add `OPENAI_API_KEY` as a secret.
4. Use persistent disk for `data/` if you want studies to survive hosting restarts. **Community Cloud local SQLite may be ephemeral and is not suitable for durable study archives.**
5. A private deployment is strongly preferred: this prototype has no login or per-user database separation.

## Workflow

1. Under **Study & Concepts**, edit the two starter concepts or add a third. Save study.
2. Under **Audience**, choose the panel and demographics. Save settings.
3. Under **Run Simulation**, enter a real API key, run batches, then select **Score all pending responses with SSR**.
4. Under **Results & Exports**, inspect distributions, underlying answers, and download data.

When changing concept wording, panel settings, or model, delete all existing responses for that study before rerunning. Otherwise saved responses will reflect mixed inputs.

## Test

```bash
python -m pytest -q
```

The unit tests do not require an API key. A full end-to-end live run requires network access, a funded API key, and the PyMC package with its model weights. No live API test is claimed in this repository.

## Code layout

- `app.py` — Streamlit UI, study workflow and dashboards
- `lab/core.py` — panels, PMF validation, aggregation, CSV shape
- `lab/generate.py` — real OpenAI response generation and image handling
- `lab/ssr.py` — published PyMC Labs `ResponseRater` adapter and disclosed prototype anchors
- `lab/store.py` — persisted SQLite studies and respondent records
- `tests/` — offline regression checks

## Primary research

Maier et al. (2025), *LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings*, https://arxiv.org/pdf/2510.08338

No Colgate-Palmolive human survey data are included in this repository.
