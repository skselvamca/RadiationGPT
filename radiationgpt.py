import streamlit as st
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import math

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="RadiationGPT",
    page_icon="☢️",
    layout="wide"
)

st.title("☢️ RadiationGPT")
st.subheader("AI Assistant for Nuclear Physics & Radiation Measurement")

# ---------------------------------------------------
# LOAD GROQ CLIENT (FIX FOR CLIENT CLOSED ERROR)
# ---------------------------------------------------

@st.cache_resource
def load_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = load_groq_client()

# ---------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# ---------------------------------------------------
# KNOWLEDGE BASE
# ---------------------------------------------------

questions = [
"What is radioactivity?",
"What is half life?",
"What is MDA?",
"What is NaI detector?",
"What is HPGe detector?",
"What is GM counter?",
"What is ZnS detector?",
"What is gamma spectroscopy?"
]

answers = [
"Radioactivity is the spontaneous emission of radiation from unstable atomic nuclei.",
"Half-life is the time required for half of radioactive atoms to decay.",
"MDA (Minimum Detectable Activity) is the lowest activity detectable with statistical confidence.",
"NaI(Tl) scintillation detector is widely used for gamma radiation detection.",
"HPGe detector is a semiconductor detector with excellent energy resolution.",
"Geiger Muller counter is a gas filled detector used for radiation survey measurements.",
"ZnS(Ag) scintillation detector is commonly used for alpha particle detection.",
"Gamma spectroscopy identifies radionuclides using characteristic gamma energies."
]

# ---------------------------------------------------
# BUILD VECTOR DATABASE
# ---------------------------------------------------

embeddings = embed_model.encode(questions)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ---------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------
# SIDEBAR DASHBOARD
# ---------------------------------------------------

st.sidebar.title("☢️ RadiationGPT Dashboard")

module = st.sidebar.radio(
"Select Module",
[
"AI Chat Assistant",
"Detector Knowledge",
"MDA Calculator",
"Activity Calculator",
"Decay Correction"
]
)

# ---------------------------------------------------
# AI CHAT MODULE
# ---------------------------------------------------

if module == "AI Chat Assistant":

    st.header("Ask RadiationGPT")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask a radiation physics question")

    if prompt:

        st.session_state.messages.append({"role":"user","content":prompt})

        with st.chat_message("user"):
            st.write(prompt)

        query_vector = embed_model.encode([prompt])
        query_vector = np.array(query_vector).astype("float32")

        D, I = index.search(query_vector, k=3)
        contexts = [answers[i] for i in I[0]]

        system_prompt = f"""
You are an expert in nuclear physics and radiation measurement.

Use this context if relevant:
{contexts}

Explain clearly and scientifically.
"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":prompt}
            ]
        )

        response = completion.choices[0].message.content

        st.session_state.messages.append({"role":"assistant","content":response})

        with st.chat_message("assistant"):
            st.write(response)

# ---------------------------------------------------
# DETECTOR KNOWLEDGE MODULE
# ---------------------------------------------------

elif module == "Detector Knowledge":

    st.header("Radiation Detector Knowledge")

    detector = st.selectbox(
        "Select Detector",
        ["NaI(Tl)", "HPGe", "GM Counter", "ZnS(Ag)"]
    )

    if detector == "NaI(Tl)":

        st.write("""
NaI(Tl) Scintillation Detector

• Used for gamma radiation detection  
• High detection efficiency  
• Moderate energy resolution  
• Common in environmental monitoring
""")

    elif detector == "HPGe":

        st.write("""
HPGe Detector

• High resolution gamma spectroscopy  
• Semiconductor detector  
• Requires liquid nitrogen cooling
""")

    elif detector == "GM Counter":

        st.write("""
Geiger-Muller Counter

• Gas filled detector  
• Detects beta and gamma radiation  
• Used in radiation survey meters
""")

    elif detector == "ZnS(Ag)":

        st.write("""
ZnS(Ag) Scintillation Detector

• Used for alpha particle detection  
• High sensitivity to alpha radiation  
• Used in gross alpha counting
""")

# ---------------------------------------------------
# MDA CALCULATOR
# ---------------------------------------------------

elif module == "MDA Calculator":

    st.header("Minimum Detectable Activity Calculator")

    B = st.number_input("Background Counts", value=100)
    t = st.number_input("Counting Time (seconds)", value=3600)
    E = st.number_input("Detector Efficiency", value=0.25)
    V = st.number_input("Sample Volume (L)", value=1.0)


    if st.button("Calculate MDA"):

        if E*t == 0:
            st.error("Efficiency and Time must be greater than zero")
        else:
            mda = (2.71 + 4.65 * math.sqrt(B)) / (E * t * V)

            st.success(f"MDA = {mda:.6f} Bq/L")


# ---------------------------------------------------
# ACTIVITY CALCULATOR
# ---------------------------------------------------

elif module == "Activity Calculator":

    st.header("Activity Calculation")

    C = st.number_input("Net Counts", value=500)
    E = st.number_input("Detector Efficiency", value=0.25)
    t = st.number_input("Counting Time (seconds)", value=3600)
    Y = st.number_input("Gamma Yield", value=0.85)
    W = st.number_input("Sample Weight (kg)", value=1.0)


    if st.button("Calculate Activity"):

        if E*t == 0:
            st.error("Efficiency and Time must be greater than zero")
        else:
            activity = C / (E * t * Y * W)

            st.success(f"Activity = {activity:.6f} Bq/kg")


# ---------------------------------------------------
# DECAY CORRECTION
# ---------------------------------------------------

elif module == "Decay Correction":

    st.header("Radioactive Decay Correction")

    A0 = st.number_input("Initial Activity (Bq)",0.0)
    t = st.number_input("Time Elapsed",0.0)
    T_half = st.number_input("Half Life",0.0)

    if st.button("Calculate Corrected Activity"):

        if T_half == 0:
            st.error("Half life must be greater than zero")
        else:
            A = A0 * math.exp(-0.693*t/T_half)
            st.success(f"Corrected Activity = {A:.6f} Bq")
