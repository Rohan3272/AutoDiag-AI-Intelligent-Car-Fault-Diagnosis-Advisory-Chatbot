# ============================================================
#  AutoDiag AI - Car Fault Diagnosis Chatbot (Web App)
#  Run with:   streamlit run app.py
# ============================================================

import re                                  # for cleaning the complaint text
import numpy as np                         # for finding the top predictions
import pandas as pd                        # for reading the knowledge base
import joblib                              # for loading our trained model
import streamlit as st                     # for building the web page
from groq import Groq                      # the AI chat library

# ---- browser tab title and icon ----
st.set_page_config(page_title="AutoDiag AI", page_icon="🚗", layout="centered")

# ---- load the model once and keep it in memory (makes the app fast) ----
@st.cache_resource
def load_everything():
    tfidf = joblib.load("tfidf_vectorizer.joblib")   # converts text into numbers
    model = joblib.load("fault_classifier.joblib")   # predicts the fault
    le    = joblib.load("label_encoder.joblib")      # number -> fault name
    kb    = pd.read_csv("knowledge_base.csv").set_index("FaultCategory")
    return tfidf, model, le, kb

tfidf, model, le, kb = load_everything()

# ---- the AI's personality and strict safety rules ----
SYSTEM_PROMPT = """You are AutoDiag AI, a friendly and caring car diagnosis assistant.
You will be given a diagnosis and verified facts from a trusted database.
Explain it to a worried car owner in simple, warm, reassuring language.

RULES:
- Only use the facts you are given. Never invent causes or advice.
- Start by telling them the most likely problem and how urgent it is.
- If it is not safe to drive, say so clearly and early.
- End by gently reminding them to see a professional mechanic.
- Keep it short and easy to understand. No technical jargon."""

# ---- clean the text exactly like we did during training ----
def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r"[^a-z0-9\s]", " ", text)     # remove symbols
    text = re.sub(r"\s+", " ", text)             # single spaces
    return text.strip()

# ---- LAYER 1: predict the top 3 most likely faults ----
def predict_faults(complaint):
    features = tfidf.transform([clean_text(complaint)])   # text -> numbers
    proba = model.predict_proba(features)[0]              # probability of each fault
    top3 = np.argsort(proba)[::-1][:3]                    # the 3 highest
    return [(le.classes_[i], proba[i]) for i in top3]

# ---- LAYER 2 + 3: look up verified facts, then let Groq write the reply ----
def get_ai_reply(complaint, client):
    top3 = predict_faults(complaint)             # LAYER 1
    fault, confidence = top3[0]                  # most likely fault
    row = kb.loc[fault]                          # LAYER 2: verified facts

    facts = f"""The user said: "{complaint}"

Most likely problem: {fault} (confidence {confidence:.0%})
Urgency level: {row['UrgencyLevel']}
Safe to drive: {row['SafeToDrive']}
Probable causes: {row['ProbableCauses']}
Precautions: {row['PrecautionarySteps']}
Recommended action: {row['RecommendedAction']}"""

    # LAYER 3: the AI writes it in friendly words
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": facts},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content, top3, row, fault, confidence

# ============================================================
#  THE WEB PAGE
# ============================================================

st.title("🚗 AutoDiag AI")
st.caption("Describe your car problem and get an instant diagnosis with safety advice.")

# ---- sidebar: API key and explanation ----
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Groq API Key", type="password",
                            help="Get a free key at console.groq.com")
    st.markdown("[Get a free key →](https://console.groq.com)")
    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. **ML model** predicts the fault\n"
        "2. **Knowledge base** gives verified safety advice\n"
        "3. **AI** explains it in simple words\n\n"
        "The AI never diagnoses on its own — it only rephrases verified facts."
    )
    st.divider()
    st.caption("Trained on 14,000 complaints across 41 fault types.")

# ---- remember the conversation ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- show the chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- the chat input box ----
if complaint := st.chat_input("e.g. my brakes squeal when I stop..."):

    # the app needs a key to work
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar first.")
        st.stop()

    # show what the user typed
    st.session_state.messages.append({"role": "user", "content": complaint})
    with st.chat_message("user"):
        st.markdown(complaint)

    # get the diagnosis and show it
    with st.chat_message("assistant"):
        with st.spinner("Diagnosing..."):
            client = Groq(api_key=api_key)
            reply, top3, row, fault, confidence = get_ai_reply(complaint, client)

        # ---- summary panel at the top ----
        # colours for the urgency and safe-to-drive badges
        urgency_colour = {"Low": "green", "Medium": "orange",
                          "High": "red", "Critical": "red"}
        safe_colour    = {"Yes": "green", "Caution": "orange", "No": "red"}

        # show the full fault name as a heading (never gets cut off)
        st.markdown(f"### 🔧 {fault}")

        # show urgency and safe-to-drive as coloured badges
        st.markdown(
            f":{urgency_colour[row['UrgencyLevel']]}[**Urgency: {row['UrgencyLevel']}**]"
            f" &nbsp;·&nbsp; :{safe_colour[row['SafeToDrive']]}[**Safe to drive: {row['SafeToDrive']}**]"
        )

        # show the confidence as a labelled progress bar
        st.progress(min(float(confidence), 1.0),
                    text=f"Model confidence in this diagnosis: {confidence:.0%}")

        # if the model is unsure, say so honestly instead of hiding it
        if confidence < 0.35:
            st.info("The description could match a few different faults, "
                    "so please also check the other possible causes below.")

        # ---- the AI's friendly explanation ----
        st.markdown(reply)

        # ---- the other possibilities ----
        with st.expander("Other possible causes", expanded=confidence < 0.35):
            for f, c in top3[1:]:
                st.write(f"- **{f}** ({c:.0%})")

    # save the reply into the conversation history
    st.session_state.messages.append({"role": "assistant", "content": reply})
