"""Streamlit UI: match resume / pasted skills to ESCO skills via a cached vector store."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from skillmatch.config import (
    DEFAULT_EMBEDDING,
    DEFAULT_RERANKER,
    EMBEDDING_MODELS,
    RERANKERS,
)
from skillmatch.embedder import DEVICE, CrossEncoderReranker, Embedder
from skillmatch.esco import load_catalog
from skillmatch.extract import phrases_from_resume, read_resume, split_skill_list
from skillmatch.llm import QwenRefiner
from skillmatch.matcher import SkillMatcher
from skillmatch.ner import JobBertExtractor
from skillmatch.store import VectorStore

st.set_page_config(page_title="ESCO Skill Matcher", page_icon="🎯", layout="wide")


NER, RULES = "JobBERT NER (recommended)", "Rule-based"


# ---------- cached resources (loaded once per session / setting) ----------
@st.cache_resource(show_spinner=False)
def get_catalog(include_alt: bool):
    return load_catalog(include_alt_labels=include_alt)


@st.cache_resource(show_spinner=False)
def get_embedder(label: str) -> Embedder:
    return Embedder(EMBEDDING_MODELS[label])


@st.cache_resource(show_spinner=False)
def get_reranker(label: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(RERANKERS[label])


@st.cache_resource(show_spinner="Loading JobBERT skill/knowledge extractors…")
def get_extractor() -> JobBertExtractor:
    return JobBertExtractor()


@st.cache_resource(show_spinner="Loading Qwen 2.5 (4-bit)…")
def get_refiner() -> QwenRefiner:
    return QwenRefiner()


@st.cache_data(show_spinner="Extracting skill spans…")
def ner_spans(text: str, refine: bool) -> list[tuple[str, str]]:
    raw = get_extractor().extract(text)
    if not refine:
        return [(sp.text, sp.kind) for sp in raw]
    refined = get_refiner().refine(text, [sp.text for sp in raw])
    return [(skill, "refined") for skill in refined]


def extract_phrases(text: str, method: str, refine: bool) -> tuple[list[str], str, dict[str, str]]:
    """Return (phrases, description, phrase -> kind) using the chosen span-extraction method."""
    if method == NER:
        spans = ner_spans(text, refine)
        return [t for t, _ in spans], "JobBERT NER" + (" + Qwen refinement" if refine else ""), dict(spans)
    phrases, strategy = phrases_from_resume(text)
    return phrases, f"rule-based: {strategy}", {}


@st.cache_resource(show_spinner=False)
def get_store(label: str, include_alt: bool) -> VectorStore:
    catalog, embedder = get_catalog(include_alt), get_embedder(label)
    if VectorStore.exists(embedder.spec, catalog):
        return VectorStore.load_or_build(embedder, catalog)
    bar = st.progress(0.0, text=f"First run for {label}: building ESCO vector store (one-time)…")
    store = VectorStore.load_or_build(embedder, catalog, progress=lambda p: bar.progress(p))
    bar.empty()
    return store


# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    emb_label = st.selectbox(
        "Embedding model",
        list(EMBEDDING_MODELS),
        index=list(EMBEDDING_MODELS).index(DEFAULT_EMBEDDING),
        help="Each model gets its own cached vector store, built once.",
    )
    st.caption(EMBEDDING_MODELS[emb_label].note)
    include_alt = st.toggle("Index alternative labels", value=True, help="~7x more entries; better recall on synonyms.")
    use_rerank = st.toggle("Cross-encoder rerank", value=True, help="Re-scores the top candidates for precision.")
    rr_label = None
    if use_rerank:
        rr_label = st.selectbox("Reranker", list(RERANKERS), index=list(RERANKERS).index(DEFAULT_RERANKER))
    st.divider()
    span_method = st.radio("Skill span extraction", [NER, RULES],
                           help="NER = SkillSpan JobBERT skill + knowledge models. Rule-based = regex / n-grams.")
    use_llm = st.toggle("Refine spans with Qwen 2.5 LLM", value=DEVICE == "cuda",
                        disabled=span_method != NER,
                        help="Cleans fragments, merges split skills and drops non-skills (slower, higher quality).")
    st.divider()
    top_k = st.slider("Matches per skill", 1, 10, 3)
    min_bi = st.slider("Min. embedding similarity", 0.0, 1.0, 0.55, 0.01,
                       help="Candidates below this are dropped before reranking (speed + precision).")
    min_final = st.slider("Min. final score", 0.0, 1.0, 0.5, 0.01)
    st.caption(f"Device: **{DEVICE.upper()}**")

st.title("🎯 ESCO Skill Matcher")
st.caption("Upload a resume or paste skills → matched against the ESCO taxonomy through a pre-built vector store.")

# ---------- input ----------
def show_spans(found: list[str], kind_of: dict[str, str], expanded: bool) -> None:
    with st.expander(f"All identified skill spans ({len(found)})", expanded=expanded):
        st.dataframe(
            pd.DataFrame({"Skill span": found, "Type": [kind_of.get(p, "") for p in found]}),
            width="stretch", hide_index=True,
        )


def extract_clicked(source: str, text: str, as_list: bool) -> None:
    """Run span extraction on demand and remember the result across reruns."""
    if as_list:
        found, how, kind_of, broad = split_skill_list(text), "split on commas / lines", {}, False
    else:
        found, how, kind_of = extract_phrases(text, span_method, use_llm)
        broad = span_method == RULES and how.endswith("n-gram scan")
    st.session_state["extraction"] = dict(
        source=source, key=(text, as_list, span_method, use_llm), phrases=found, how=how, kinds=kind_of, broad=broad
    )
    st.session_state.pop("result", None)  # results from older spans are stale


tab_resume, tab_skills = st.tabs(["📄 Upload resume", "✍️ Paste skills"])
with tab_resume:
    f = st.file_uploader("Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    resume_text = read_resume(f.name, f.getvalue()) if f else ""
    if f:
        with st.expander("Extracted text"):
            st.text(resume_text[:5000])
    st.button("Extract skill spans", key="extract_resume", disabled=not resume_text.strip(),
              on_click=extract_clicked, args=("resume", resume_text, False))
with tab_skills:
    mode = st.radio("Input is", ["A list of skills", "Free text (sentences)"], horizontal=True)
    pasted = st.text_area("Text", height=180,
                          placeholder="python programming, project management, customer service, SQL")
    st.button("Extract skill spans", key="extract_pasted", disabled=not pasted.strip(),
              on_click=extract_clicked, args=("pasted", pasted, mode == "A list of skills"))

ext = st.session_state.get("extraction")
phrases: list[str] = ext["phrases"] if ext else []
broad_scan = bool(ext and ext["broad"])
if ext:
    st.caption(f"Extracted **{len(phrases)}** skill spans from the {ext['source']} ({ext['how']}).")
    show_spans(phrases, ext["kinds"], expanded=len(phrases) <= 60)

run = st.button("Match skills", type="primary", disabled=not phrases)

# ---------- run ----------
if run:
    with st.spinner("Loading models and vector store…"):
        store = get_store(emb_label, include_alt)
        matcher = SkillMatcher(get_embedder(emb_label), store, get_reranker(rr_label) if rr_label else None)
    # a broad n-gram scan produces lots of junk -> use a stricter retrieval cut-off
    with st.spinner(f"Matching {len(phrases)} phrases…"):
        result = matcher.match(
            phrases,
            top_k=top_k,
            min_bi_score=max(min_bi, 0.7) if broad_scan else min_bi,
            min_final_score=min_final,
        )
    st.session_state["result"] = result
    st.session_state["phrases"] = phrases

result = st.session_state.get("result")
if result:
    t = result.timings
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Input phrases", result.n_queries)
    c2.metric("Unique ESCO skills", len(result.matches))
    c3.metric("Total time", f"{t['total'] * 1000:.0f} ms")
    c4.metric("Embed / search / rerank",
              f"{t['embed'] * 1000:.0f} / {t['search'] * 1000:.0f} / {t['rerank'] * 1000:.0f} ms")

    # Side-by-side summary: final candidate skills (inputs that matched) vs. resulting ESCO skills.
    matched_inputs = sorted(result.per_query, key=str.lower)
    unmatched = [p for p in st.session_state.get("phrases", []) if p not in result.per_query]
    left, right = st.columns(2)
    with left:
        st.subheader(f"Candidate skills ({len(matched_inputs)})")
        st.markdown(" ".join(f"`{s}`" for s in matched_inputs) or "_none_")
        if unmatched and len(unmatched) <= 100:
            with st.expander(f"No match above threshold ({len(unmatched)})"):
                st.write(", ".join(unmatched))
    with right:
        st.subheader(f"ESCO skills ({len(result.matches)})")
        st.markdown(" ".join(f"`{m.label}`" for m in result.matches) or "_none_")
    st.divider()

    view = st.radio("View", ["Unique ESCO skills", "Per input skill"], horizontal=True)
    if view == "Unique ESCO skills":
        df = pd.DataFrame(
            [
                {"ESCO skill": m.label, "Matched from": m.query, "Score": round(m.score, 3),
                 "Embedding": round(m.bi_score, 3),
                 "Rerank": None if m.rerank_score is None else round(m.rerank_score, 3),
                 "Type": m.skill_type, "Description": m.description, "URI": m.uri}
                for m in result.matches
            ]
        )
        st.dataframe(df, width='stretch', hide_index=True,
                     column_config={"Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                    "URI": st.column_config.LinkColumn()})
        st.download_button("Download CSV", df.to_csv(index=False), "esco_matches.csv", "text/csv")
    else:
        for q, ms in result.per_query.items():
            with st.expander(f"{q}  →  {ms[0].label}  ({ms[0].score:.2f})"):
                for m in ms:
                    st.markdown(f"**{m.label}** · score {m.score:.2f} · {m.skill_type}  \n{m.description}")
