# ============================================================
#  AutoDiag AI - Polished Web App
#  Run with:   streamlit run app_pro.py
# ============================================================

import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from groq import Groq

st.set_page_config(page_title="AutoDiag AI", page_icon="🏎️",
                   layout="wide", initial_sidebar_state="expanded")  # always open

@st.cache_resource
def load_everything():
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    model = joblib.load("fault_classifier.joblib")
    le    = joblib.load("label_encoder.joblib")
    kb    = pd.read_csv("knowledge_base.csv").set_index("FaultCategory")
    return tfidf, model, le, kb

tfidf, model, le, kb = load_everything()

# ============================================================
#  STYLING
# ============================================================
st.markdown("""
<style>
.stApp {
    background: #0f1117;
    color: #e6e8ec;
}
#MainMenu, header, footer {visibility: hidden;}

/* ---- title ---- */
.hero-title {
    font-size: 2.1rem; font-weight: 700; text-align: center;
    margin: 0.2rem 0 0.1rem 0; color: #f2f3f5; letter-spacing: -0.5px;
}
.hero-title span { color: #e2603c; }
.hero-sub { text-align:center; color:#8b909a; font-size:0.96rem; margin-bottom:1.6rem; }

/* ---- subtle fade-in ---- */
@keyframes fadeUp { from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:translateY(0)} }
.fade { animation: fadeUp 0.35s ease both; }

/* ---- diagnosis card ---- */
.diag-card {
    background: #171a22;
    border: 1px solid #262b36; border-left: 3px solid #e2603c;
    border-radius: 12px; padding: 18px 20px; margin: 8px 0 4px 0;
}
.diag-fault { font-size: 1.3rem; font-weight: 650; color: #f2f3f5; margin-bottom: 12px; }
.diag-label { color:#8b909a; font-size:0.8rem; margin-bottom:5px; }

/* ---- badges ---- */
.badge {
    display:inline-block; padding:4px 12px; border-radius:6px;
    font-size:0.8rem; font-weight:600; margin-right:8px;
}
.b-red    { background:rgba(224,80,80,0.13);  color:#e57373; border:1px solid rgba(224,80,80,0.3);}
.b-orange { background:rgba(224,150,60,0.13); color:#e0a45c; border:1px solid rgba(224,150,60,0.3);}
.b-green  { background:rgba(80,190,120,0.13); color:#66bb8a; border:1px solid rgba(80,190,120,0.3);}

/* ---- confidence bar ---- */
.conf-track { background:#20242e; border-radius:6px; height:7px; overflow:hidden; margin:5px 0; }
.conf-fill  { height:100%; border-radius:6px; background:#e2603c; transition:width 0.7s ease; }

/* ---- reply text ---- */
.reply-box {
    background:#13161d; border:1px solid #232833; border-radius:12px;
    padding:16px 18px; margin-top:10px; line-height:1.65; color:#d4d8df; font-size:0.96rem;
}

/* ---- example buttons ---- */
div[data-testid="stButton"] > button {
    background:#161a22; color:#b8bdc7; border:1px solid #262b36;
    border-radius:10px; padding:11px 14px; font-size:0.9rem; text-align:left;
    transition:all 0.15s ease; width:100%; font-weight:400;
}
div[data-testid="stButton"] > button:hover {
    border-color:#e2603c; color:#f2f3f5; background:#1a1f28;
}

/* ---- primary button (Start) : dim by default, lit when enabled ---- */
div[data-testid="stButton"] > button[kind="primary"] {
    background:#e2603c; color:#fff; border:none; font-weight:600;
    transition:all 0.2s ease;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background:#f0714d; box-shadow:0 4px 16px rgba(226,96,60,0.35);
}
div[data-testid="stButton"] > button[kind="primary"]:disabled {
    background:#2a2e38; color:#6b7280; cursor:not-allowed;
}

/* ---- clean text input (no overlap) ---- */
.stTextInput > div > div > input {
    background:#171a22 !important; color:#f2f3f5 !important;
    border:1px solid #2a3040 !important; border-radius:10px !important;
    padding:12px 14px !important; font-size:0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color:#e2603c !important; box-shadow:0 0 0 2px rgba(226,96,60,0.15) !important;
}
/* hide Streamlit's "Press Enter to apply" helper that overlaps the field */
.stTextInput div[data-testid="InputInstructions"] { display:none !important; }
[data-testid="InputInstructions"] { display:none !important; }

/* ---- sidebar history items ---- */
.hist-item {
    padding:9px 12px; border-radius:8px; margin-bottom:5px; cursor:default;
    background:#161a22; border:1px solid #222732; color:#c2c7d0; font-size:0.85rem;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
section[data-testid="stSidebar"] {
    background:#0b0d12; border-right:1px solid #1c212b;
    min-width:290px !important; max-width:290px !important;
    transform:none !important; visibility:visible !important;
}
/* remove the collapse arrow completely so the sidebar can never be hidden */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display:none !important; }

.gate-wrap { max-width:460px; margin:4rem auto 0 auto; }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  LOGIC
# ============================================================
SYSTEM_PROMPT = """You are AutoDiag AI, a friendly and caring car diagnosis assistant.
You will be given a diagnosis and verified facts from a trusted database.
Explain it to a worried car owner in simple, warm, reassuring language.
RULES:
- Only use the facts you are given. Never invent causes or advice.
- Start by telling them the most likely problem and how urgent it is.
- If it is not safe to drive, say so clearly and early.
- End by gently reminding them to see a professional mechanic.
- Keep it short and easy to understand. No technical jargon."""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_faults(complaint):
    features = tfidf.transform([clean_text(complaint)])
    proba = model.predict_proba(features)[0]
    top3 = np.argsort(proba)[::-1][:3]
    return [(le.classes_[i], proba[i]) for i in top3]

def get_ai_reply(complaint, client):
    top3 = predict_faults(complaint)
    fault, confidence = top3[0]
    row = kb.loc[fault]
    facts = f"""The user said: "{complaint}"

Most likely problem: {fault} (confidence {confidence:.0%})
Urgency level: {row['UrgencyLevel']}
Safe to drive: {row['SafeToDrive']}
Probable causes: {row['ProbableCauses']}
Precautions: {row['PrecautionarySteps']}
Recommended action: {row['RecommendedAction']}"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": facts}],
        temperature=0.7,
    )
    return response.choices[0].message.content, top3, row, fault, confidence

# ============================================================
#  API KEY GATE
# ============================================================
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if not st.session_state.api_key:
    st.markdown('<div class="gate-wrap fade">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Auto<span>Diag</span> AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Intelligent car fault diagnosis assistant</div>',
                unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Enter your Groq API key to begin**")
        # a form lets the button reliably enable and submit together
        with st.form("key_form", border=False):
            key_input = st.text_input("key", type="password", placeholder="gsk_...",
                                      label_visibility="collapsed")
            submitted = st.form_submit_button("Start Diagnosing", type="primary",
                                              use_container_width=True)
        st.caption("Don't have a key? [Get one free at console.groq.com](https://console.groq.com) · Your key stays in this browser session only.")

        if submitted:
            if key_input.strip():
                st.session_state.api_key = key_input.strip()
                st.rerun()
            else:
                st.error("Please paste your Groq API key first.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ============================================================
#  MAIN APP
# ============================================================
if "chats" not in st.session_state:
    st.session_state.chats = []       # list of {"q":..., "fault":..., "reply":..., ...}
if "pending" not in st.session_state:
    st.session_state.pending = None

# ---- SIDEBAR : ChatGPT-style history ----
with st.sidebar:
    st.markdown("### 🏎️ AutoDiag AI")
    if st.button("＋  New diagnosis", use_container_width=True):
        st.session_state.chats = []
        st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**History**")
    if st.session_state.chats:
        for ch in reversed(st.session_state.chats):
            st.markdown(f'<div class="hist-item">{ch["q"][:38]}</div>', unsafe_allow_html=True)
    else:
        st.caption("Your past questions in this session will appear here.")
    st.divider()
    if st.button("Change API key", use_container_width=True):
        st.session_state.api_key = ""
        st.rerun()

# ---- HEADER ----
st.markdown('<div class="hero-title">Auto<span>Diag</span> AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Describe your car problem and get an instant diagnosis with safety advice</div>',
            unsafe_allow_html=True)

# ---- example prompts (only on a fresh screen) ----
if not st.session_state.chats:
    st.markdown("###### Try an example, or type your own below")
    examples = [
        "Brakes squeal every time I press them",
        "Brake pedal feels soft and goes to the floor",
        "Car won't start, just a clicking sound",
        "Temperature gauge in the red and steam from the bonnet",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, key=f"ex{i}", use_container_width=True):
                st.session_state.pending = ex

# ---- render all past exchanges (ChatGPT-style transcript) ----
for ch in st.session_state.chats:
    with st.chat_message("user"):
        st.markdown(ch["q"])
    with st.chat_message("assistant"):
        st.markdown(ch["card"], unsafe_allow_html=True)
        st.markdown(f'<div class="reply-box">{ch["reply"]}</div>', unsafe_allow_html=True)
        if ch["alts"]:
            with st.expander("Other possible causes"):
                for f, c in ch["alts"]:
                    st.markdown(f"- **{f}** &nbsp;·&nbsp; {c:.0%}")

# ---- input ----
typed = st.chat_input("Describe the problem, e.g. grinding noise when I brake...")
complaint = st.session_state.pop("pending", None) or typed

if complaint:
    with st.chat_message("user"):
        st.markdown(complaint)

    with st.chat_message("assistant"):
        with st.spinner("Diagnosing..."):
            client = Groq(api_key=st.session_state.api_key)
            reply, top3, row, fault, confidence = get_ai_reply(complaint, client)

        urg = row["UrgencyLevel"]; safe = row["SafeToDrive"]
        urg_cls = "b-red" if urg in ("High", "Critical") else ("b-orange" if urg == "Medium" else "b-green")
        safe_cls = "b-green" if safe == "Yes" else ("b-orange" if safe == "Caution" else "b-red")
        conf_pct = int(confidence * 100)

        card = f"""
<div class="diag-card fade">
  <div class="diag-fault">{fault}</div>
  <div class="diag-label">Diagnosis confidence &nbsp;{conf_pct}%</div>
  <div class="conf-track"><div class="conf-fill" style="width:{conf_pct}%"></div></div>
  <div style="margin-top:12px">
    <span class="badge {urg_cls}">Urgency: {urg}</span>
    <span class="badge {safe_cls}">Safe to drive: {safe}</span>
  </div>
</div>
"""
        st.markdown(card, unsafe_allow_html=True)
        if confidence < 0.35:
            st.info("The description could match a few different faults - please also check the alternatives below.")
        st.markdown(f'<div class="reply-box fade">{reply}</div>', unsafe_allow_html=True)
        alts = [(f, c) for f, c in top3[1:]]
        with st.expander("Other possible causes"):
            for f, c in alts:
                st.markdown(f"- **{f}** &nbsp;·&nbsp; {c:.0%}")

    # save to history so it renders as a transcript on rerun
    st.session_state.chats.append({
        "q": complaint, "card": card, "reply": reply,
        "fault": fault, "alts": alts,
    })
    st.rerun()
