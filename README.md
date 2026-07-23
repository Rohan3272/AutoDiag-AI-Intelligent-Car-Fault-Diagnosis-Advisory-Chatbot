# AutoDiag AI — Intelligent Car Fault Diagnosis & Advisory Chatbot

Describe a car problem in plain language, and this system tells you the most likely fault, how urgent it is, whether the car is safe to drive, and what to do next.

**Capstone Project 2** · Rohan Mehta

---

## What it does

You type something like *"my brake pedal feels soft and goes almost to the floor"* and the system responds with a diagnosis (Brake Fluid Leak), an urgency level (Critical), a safe-to-drive verdict (No), the probable causes, and the precautions to take, written in plain, calm language.

## How it works: three layers

| Layer | Role |
|---|---|
| **1. Prediction** | A trained ML model classifies the complaint into one of 41 fault categories and returns a confidence score |
| **2. Knowledge grounding** | The predicted fault is looked up in a curated knowledge base holding verified causes, precautions, urgency and safe-to-drive status |
| **3. Conversation** | The Groq API (Llama 3.3) rewrites those verified facts as a warm, human reply |

**The key safety design:** the language model is never allowed to diagnose or invent advice. It only rephrases facts supplied by the knowledge base, so every safety instruction traces back to a checked source.

---

## The dataset

- **14,350 records** across 19 fields (14,000 unique complaints)
- **41 fault categories** across 9 vehicle systems
- Synthetic, but structurally grounded in **SAE J2012 / OBD-II** standard trouble codes, OEM service-manual chapter structure, public automotive-forum phrasing patterns, and **NHTSA / AAA** safety guidance
- Deliberately contains **12 categories of real-world data-quality problems**: six date formats, mixed currency and mileage encodings, brand typos, seven disguised null styles, impossible outliers, and duplicate records

> The data is synthetic because complaint-level workshop data is proprietary. The pipeline, methodology and architecture transfer directly to real data; the reported figures describe this dataset, not the real world.

---

## Results

Six models were trained on identical stratified splits:

| Model | Accuracy | Macro-F1 | Top-3 |
|---|---|---|---|
| Naive Bayes | 0.872 | 0.872 | 0.929 |
| Logistic Regression | 0.871 | 0.871 | 0.950 |
| Linear SVM | 0.870 | 0.870 | — |
| DistilBERT (fine-tuned) | ~0.87 | ~0.87 | ~0.95 |
| BiLSTM | ~0.86 | ~0.86 | ~0.94 |
| Random Forest | 0.865 | 0.865 | 0.947 |

**All six converge around 87%**, which is a finding rather than a failure: the ceiling is the data, not the model. Complaints are short (median 12 words) and keyword-driven, so simple word features already capture nearly all the available signal. The residual error is genuine ambiguity between mechanically similar faults, such as *Clogged Cabin Filter vs AC Compressor Failure* or *Alternator vs Wiring Fault*, which a human mechanic would also need to inspect the car to separate.

For context: random guessing across 41 classes scores **2.4%**.

**Logistic Regression is deployed** rather than DistilBERT, because the application needs calibrated probabilities, millisecond CPU inference, and interpretability.

---

## Repository structure

```
Part1/    Pipeline notebook, trained model files, knowledge base
Part2/    Streamlit chatbot app and explanatory chatbot notebook
Raw/      Raw dataset and knowledge base
```

---

## Running it

**1. Install the requirements**

```bash
pip install -r Part2/requirements.txt
```

**2. Get a free Groq API key** at [console.groq.com](https://console.groq.com)

**3. Run the web app**

```bash
cd Part2
streamlit run app.py
```

Paste your API key into the sidebar, then describe a car problem in the chat box.

**4. Or explore the pipeline**

Open `Part1/Complete_Pipeline_AutoDiag.ipynb` to see the full workflow: cleaning, EDA, transformation, model building, tuning, and evaluation.

> **Note:** the pipeline notebook uses absolute file paths. Update the `read_csv` paths at the top to match wherever you have cloned the repository.

---

## Tech stack

Python · Pandas · NumPy · Scikit-learn · TensorFlow/Keras · Hugging Face Transformers · Matplotlib · Seaborn · WordCloud · Streamlit · Groq API (Llama 3.3 70B)

---

## Limitations

- The dataset is synthetic and would need validation on real workshop data before production use
- The system works only from what the owner describes, so it cannot detect anything they have not noticed
- It is a triage tool, not a replacement for a mechanic. Every response recommends professional inspection
