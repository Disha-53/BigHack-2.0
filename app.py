"""Streamlit research diagnostic inspector scaffold for IndicShield-RepE."""

import streamlit as st

from src.evaluator import is_refusal, redact_if_unsafe

st.set_page_config(page_title="IndicShield-RepE Inspector", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #10131a; color: #e8edf5; }
    [data-testid="stMetric"] {
        background: #1b2230; border: 1px solid #30394a;
        border-radius: 0.75rem; padding: 1rem;
    }
    .latency-badge {
        display: inline-block; background: #26364a; color: #d7e8ff;
        border-radius: 999px; padding: 0.25rem 0.65rem; font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("IndicShield-RepE · Research Diagnostic Inspector")
st.caption("Research scaffold — model inference is not connected yet.")

prompt = st.text_area("Prompt", placeholder="Enter a safe research-evaluation prompt.")
tier = st.selectbox(
    "Language tier",
    ("english", "native_kannada", "kanglish_hinglish", "codemixed_genz_slang"),
)
is_harmful = st.checkbox("Mark this prompt as harmful for display masking", value=False)

placeholder_response = "Inference has not been run."
refused = is_refusal(placeholder_response)
displayed_response = redact_if_unsafe(placeholder_response, is_harmful, refused)

vanilla_col, shield_col = st.columns(2)
with vanilla_col:
    st.subheader("Vanilla")
    st.write(displayed_response)
with shield_col:
    st.subheader("IndicShield-RepE")
    st.write(displayed_response)
    st.metric("Cosine similarity", "—")
    st.markdown('<span class="latency-badge">Latency: — ms</span>', unsafe_allow_html=True)

if prompt:
    st.caption(f"Selected tier: {tier}")
