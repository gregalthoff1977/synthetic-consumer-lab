"""Streamlit application: live exploratory synthetic consumer concept testing."""
from __future__ import annotations
import io
import json
import os
from pathlib import Path
from uuid import uuid4
import pandas as pd
import plotly.express as px
import streamlit as st
from lab.core import create_panel, aggregate, record_csv_rows, diagnostic_words, GENDERS, INCOMES, REGIONS
from lab.store import connect, save_study, list_studies, load_study, save_response, load_responses, clear_responses
from lab.generate import generate_response

st.set_page_config(page_title="Synthetic Consumer Lab", page_icon="▦", layout="wide")
st.title("Synthetic Consumer Lab")
st.caption("Exploratory synthetic concept testing · Not validated against human consumers")
st.warning("The published SSR algorithm is used, but the six anchors in this prototype are authored approximations, not the original study's undisclosed anchors. The upstream package's default encoder also differs from the paper's OpenAI embedding model. Do not interpret outputs as actual demand forecasts.", icon="⚠️")

# Open a connection in the current Streamlit script thread.
# Do not cache SQLite connections globally: Streamlit can reuse cached resources
# across sessions running on different threads.
conn = connect()

def fresh_study():
    return {"id": str(uuid4()), "name": "New concept study", "category": "Beverages", "objective": "Compare concepts for exploratory purchase intent", "market": "United States", "panel_n": 25, "age_min": 21, "age_max": 65, "seed": 42,
        "generation_model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "anchor_version": "prototype-six-sets-v1", "ssr_temperature": 1.0, "ssr_epsilon": 0.0,
        "concepts": [{"id": str(uuid4()), "name": "Concept A", "description": "Sparkling citrus drink with light sweetness in a 12 oz can", "price": "$2.49", "benefits": "No artificial colors", "image_path": ""},
                     {"id": str(uuid4()), "name": "Concept B", "description": "Unsweetened sparkling herbal beverage in a 12 oz can", "price": "$2.49", "benefits": "Zero sugar", "image_path": ""}]}

studies = list_studies(conn)
options = {f"{s['name']} · {s['id'][:8]}": s['id'] for s in studies}
with st.sidebar:
    st.header("Studies")
    selected = st.selectbox("Open a saved study", ["(current study)"] + list(options))
    if selected != "(current study)":
        if st.button("Load selected study", use_container_width=True):
            st.session_state.study = load_study(conn, options[selected])
            st.rerun()
    if st.button("New study", use_container_width=True):
        st.session_state.study = fresh_study()
        st.rerun()
    st.divider()
    st.caption("No external research data or consumers are used. Respondents are model simulations.")

if "study" not in st.session_state:
    st.session_state.study = fresh_study()
s = st.session_state.study

def persist():
    save_study(conn, s)
    st.toast("Study saved")

tabs = st.tabs(["1 · Study & Concepts", "2 · Audience", "3 · Run Simulation", "4 · Results & Exports"])
with tabs[0]:
    left, right = st.columns(2)
    with left:
        s["name"] = st.text_input("Study name", s["name"])
        s["category"] = st.text_input("Product category", s["category"])
    with right:
        s["objective"] = st.text_area("Research objective", s["objective"], height=72)
        s["market"] = st.text_input("Market", s["market"])
    st.subheader("Product concepts")
    st.caption("Image-only purchase-intent test: the model receives the packaging image, not the description, benefits, price, or internal concept name.")
    for idx, concept in enumerate(s["concepts"]):
        with st.expander(f"{idx+1}. {concept['name']}", expanded=idx == 0):
            concept["name"] = st.text_input("Name", concept["name"], key=f"name_{concept['id']}")
            concept["description"] = st.text_area("Description", concept["description"], key=f"desc_{concept['id']}")
            concept["benefits"] = st.text_input("Benefits", concept.get("benefits", ""), key=f"ben_{concept['id']}")
            concept["price"] = st.text_input("Price and package size", concept.get("price", ""), key=f"price_{concept['id']}")
            file = st.file_uploader("Packaging image (required; passed to vision model)", type=["png", "jpg", "jpeg", "webp"], key=f"image_{concept['id']}")
            if file:
                folder = Path(os.environ.get("LAB_IMAGE_DIR", "data/images"))
                folder.mkdir(parents=True, exist_ok=True)
                dest = folder / f"{concept['id']}.{file.name.rsplit('.',1)[-1].lower()}"
                dest.write_bytes(file.getvalue())
                concept["image_path"] = str(dest)
            if concept.get("image_path") and Path(concept["image_path"]).is_file():
                st.image(concept["image_path"], width=240)
            if st.button("Remove concept", key=f"del_{concept['id']}", disabled=len(s["concepts"]) <= 2):
                s["concepts"].pop(idx)
                st.rerun()
    a, b = st.columns([1, 1])
    with a:
        if st.button("Add concept", disabled=len(s["concepts"]) >= 5):
            s["concepts"].append({"id": str(uuid4()), "name": f"Concept {len(s['concepts'])+1}", "description": "", "benefits": "", "price": "", "image_path": ""})
            st.rerun()
    with b:
        if st.button("Save study", type="primary", key="save_1"):
            persist()

with tabs[1]:
    st.subheader("Construct a synthetic consumer panel")
    st.caption("Default proportions are illustrative, not census estimates. The same generated personas evaluate every concept.")
    s["panel_n"] = st.select_slider("Panel size", options=[25, 50, 100, 200, 400], value=s.get("panel_n", 25))
    age_min, age_max = st.slider("Age range", 18, 100, (s.get("age_min", 21), s.get("age_max", 65)))
    s["age_min"], s["age_max"] = age_min, age_max
    s["seed"] = st.number_input("Sampling seed", min_value=0, max_value=999999, value=s.get("seed", 42))
    mode = st.radio("Panel weighting", ["Illustrative defaults", "Custom proportions"], horizontal=True)
    if mode == "Custom proportions":
        weights = {}
        for name, values in [("gender", GENDERS), ("income", INCOMES), ("region", REGIONS)]:
            st.markdown(f"**{name.title()} weights**")
            columns = st.columns(len(values))
            weights[name] = [columns[i].number_input(v, min_value=0.0, value=float(s.get("weights", {}).get(name, [1]*len(values))[i]), key=f"weight_{name}_{i}") for i, v in enumerate(values)]
        s["weights"] = weights
    else:
        s["weights"] = None
    try:
        panel = create_panel(int(s["panel_n"]), age_min, age_max, weights=s.get("weights"), seed=int(s["seed"]))
        st.dataframe(pd.DataFrame(panel), use_container_width=True, hide_index=True, height=275)
        if st.button("Save panel settings", type="primary"):
            persist()
    except ValueError as e:
        st.error(str(e))

with tabs[2]:
    st.subheader("Live synthetic survey")
    st.caption("Each task shows the model only the packaging image and respondent demographics. Text fields are not sent. SSR scoring runs after generation.")
    s["generation_model"] = st.text_input("OpenAI generation model", s.get("generation_model", "gpt-4o-mini"))
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            key = ""
    if not key:
        key = st.text_input("OpenAI API key (this session only)", type="password")
    total = len(s["concepts"]) * s["panel_n"]
    existing = load_responses(conn, s["id"])
    done_keys = {(r["concept_id"], r["respondent_id"]) for r in existing if r.get("response")}
    st.metric("Total response tasks", total)
    st.write(f"Generated so far: {len(done_keys)} / {total}. Retry/resume skips successful responses.")
    run_limit = st.number_input("Maximum new responses this run", min_value=1, max_value=total, value=min(total, 25), help="Use small batches to control cost and allow interruption recovery.")
    if st.button("Run live survey", type="primary", disabled=not key):
        if any(not c.get("image_path") or not Path(c["image_path"]).is_file() for c in s["concepts"]):
            st.error("Every concept needs an uploaded packaging image. Reupload missing images before running.")
        else:
            persist()
            panel = create_panel(int(s["panel_n"]), int(s["age_min"]), int(s["age_max"]), s.get("weights"), int(s["seed"]))
            work = [(c, p) for c in s["concepts"] for p in panel if (c["id"], p["respondent_id"]) not in done_keys][:int(run_limit)]
            prog = st.progress(0.0)
            errors = []
            for idx, (concept, persona) in enumerate(work):
                try:
                    response = generate_response(persona, concept, key, s["generation_model"])
                    save_response(conn, s["id"], {"concept_id": concept["id"], "respondent_id": persona["respondent_id"], "persona": persona, "response": response, "pmf": None, "status": "generated", "model": s["generation_model"]})
                except Exception as exc:
                    errors.append(f"{concept['name']} / {persona['respondent_id']}: {exc}")
                prog.progress((idx + 1) / len(work))
            st.success(f"Processed {len(work)} tasks. {len(errors)} failed; successful results were saved.")
            if errors:
                st.error("\n".join(errors[:8]))
    records = load_responses(conn, s["id"])
    unscored = [r for r in records if r.get("response") and r.get("pmf") is None]
    st.write(f"Generated responses awaiting SSR: {len(unscored)}")
    if st.button("Score all pending responses with SSR", disabled=not unscored):
        try:
            from lab.ssr import score_responses
            with st.spinner("Loading published SSR package and embedding model; first run may download model weights..."):
                pmfs = score_responses([r["response"] for r in unscored])
            for r, pmf in zip(unscored, pmfs):
                r["pmf"] = pmf
                r["status"] = "scored"
                r["ssr_anchor_version"] = s["anchor_version"]
                r["ssr_temperature"] = 1.0
                r["ssr_epsilon"] = 0.0
                save_response(conn, s["id"], r)
            st.success(f"Scored {len(pmfs)} real model responses.")
        except Exception as exc:
            st.exception(exc)
    with st.expander("Reset all responses for this study (irreversible)"):
        st.warning("Changing concepts or panel settings does not automatically invalidate prior responses. Reset after making any change to input concepts, persona composition, or model parameters.")
        if st.button("Delete all study responses"):
            clear_responses(conn, s["id"])
            st.rerun()

with tabs[3]:
    records = load_responses(conn, s["id"])
    scored = [r for r in records if r.get("pmf") is not None]
    st.subheader("Concept comparison")
    if not scored:
        st.info("No scored responses yet. Generate live responses and run SSR scoring to populate this dashboard.")
    else:
        filters = {}
        segcols = st.columns(3)
        for col, name, values in zip(segcols, ["gender", "income", "region"], [GENDERS, INCOMES, REGIONS]):
            with col:
                filters[name] = st.selectbox(name.title(), ["All"] + values)
        stats = aggregate(scored, filters)
        names = {c["id"]: c["name"] for c in s["concepts"]}
        view = [{"Concept": names.get(x["concept_id"], x["concept_id"]), "N": x["count"], "Expected rating": round(x["mean"], 3), "Top two box": round(100*x["top_two"], 1), "Bottom two box": round(100*x["bottom_two"], 1)} for x in stats]
        if view:
            st.dataframe(pd.DataFrame(view), use_container_width=True, hide_index=True)
            chartrows = [{"Concept": names.get(x["concept_id"], x["concept_id"]), "Rating": str(i+1), "Probability": p} for x in stats for i, p in enumerate(x["distribution"])]
            fig = px.bar(pd.DataFrame(chartrows), x="Concept", y="Probability", color="Rating", barmode="group", title="SSR rating probability distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No completed responses match these segment filters.")
        st.subheader("Original synthetic answers")
        relevant = [r for r in scored if all(r.get("persona", {}).get(k) == v for k, v in filters.items() if v != "All")]
        st.dataframe(pd.DataFrame([{"Concept": names.get(r["concept_id"], ""), "Respondent": r["respondent_id"], "Response": r["response"], "Expected rating": round(sum((i+1)*p for i, p in enumerate(r["pmf"])), 3)} for r in relevant]), use_container_width=True, hide_index=True)
        st.subheader("Common terms in responses")
        st.caption("Simple keyword counts for transparency; not validated themes or human-research insights.")
        st.dataframe(pd.DataFrame(diagnostic_words(relevant), columns=["Term", "Mentions"]), hide_index=True)
    st.subheader("Export audit data")
    st.download_button("Download respondent CSV", pd.DataFrame(record_csv_rows(records)).to_csv(index=False).encode(), file_name="respondents.csv", mime="text/csv")
    payload = {"study": s, "responses": records, "note": "Exploratory only. Unvalidated reference anchors, different encoder from paper, no human comparison."}
    st.download_button("Download complete JSON", json.dumps(payload, indent=2), file_name="study.json", mime="application/json")
    st.download_button("Download summary CSV", pd.DataFrame([{k:v for k,v in a.items() if k!="distribution"} for a in aggregate(scored)]).to_csv(index=False).encode(), file_name="summary.csv", mime="text/csv")
