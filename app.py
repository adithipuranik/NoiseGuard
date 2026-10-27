

import streamlit as st

import numpy as np

import pandas as pd

import librosa

import librosa.display

import matplotlib.pyplot as plt

import joblib

import tempfile

import os
import io

from datetime import datetime

# =========================================================

# PAGE CONFIGURATION

# =========================================================

st.set_page_config(

    page_title="NoiseGuard",


    layout="wide"

)

# =========================================================

# LOAD TRAINED MODEL

# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "noiseguard_final_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "noiseguard_final_label_encoder.pkl")

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(ENCODER_PATH)

# =========================================================

# ENVIRONMENTAL SOUND CATEGORIES

# =========================================================

sound_categories = {

    "air_conditioner": "Building / Mechanical Noise",

    "car_horn": "Traffic Noise",

    "children_playing": "Human Activity",

    "dog_bark": "Animal / Residential Sound",

    "drilling": "Construction Noise",

    "engine_idling": "Vehicle / Mechanical Noise",

    "gun_shot": "Sudden High-Intensity Sound",

    "jackhammer": "Construction Noise",

    "siren": "Emergency / Traffic Sound",

    "street_music": "Public / Cultural Activity"

}

# =========================================================

# MONITORING SESSION STORAGE

# =========================================================

if "monitoring_history" not in st.session_state:

    st.session_state.monitoring_history = []

# =========================================================

# FEATURE EXTRACTION
