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
    page_icon="🎧",
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
# MUST MATCH TRAINING EXACTLY
# =========================================================
def extract_rich_features(file_path, n_mfcc=40):
    audio, sample_rate = librosa.load(
        file_path,
        sr=None
    )
    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc
    )
    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate
    )
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate
    )
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate
    )
    # Zero-crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        audio
    )
    # 112-feature representation used during training
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        np.mean(spectral_centroid, axis=1),
        np.std(spectral_centroid, axis=1),
        np.mean(spectral_bandwidth, axis=1),
        np.std(spectral_bandwidth, axis=1),
        np.mean(spectral_rolloff, axis=1),
        np.std(spectral_rolloff, axis=1),
        np.mean(zero_crossing_rate, axis=1),
        np.std(zero_crossing_rate, axis=1)
    ])
    return features
# =========================================================
# ACOUSTIC ACTIVITY ANALYSIS
# =========================================================
def analyze_acoustic_activity(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=None
    )
    rms = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=512
    )[0]
    rms_db = librosa.amplitude_to_db(
        rms,
        ref=np.max
    )
    # Relative threshold within the recording
    threshold = np.percentile(
        rms_db,
        50
    )
    active = rms_db >= threshold
    active_percentage = (
        np.mean(active) * 100
    )
    duration = len(audio) / sr
    mean_rms = float(
        np.mean(rms)
    )
    max_rms = float(
        np.max(rms)
    )
    if active_percentage < 40:
        activity = "Low"
    elif active_percentage < 70:
        activity = "Moderate"
    else:
        activity = "High"
    return {
        "duration": round(duration, 2),
        "mean_rms": round(mean_rms, 4),
        "max_rms": round(max_rms, 4),
        "active_percentage": round(
            float(active_percentage),
            2
        ),
        "activity": activity
    }
# =========================================================
# ENVIRONMENTAL ASSESSMENT
# =========================================================
def generate_assessment(
    predicted_class,
    category,
    activity
):
    if activity == "High":
        return (
            f"High relative acoustic activity detected. "
            f"The identified source is **"
            f"{predicted_class.replace('_', ' ').title()}**, "
            f"belonging to the **{category}** category. "
            f"This recording should be considered a "
            f"high-activity monitoring event. Repeated "
            f"recordings can help determine whether the "
            f"source is persistent."
        )
    elif activity == "Moderate":
        return (
            f"Moderate relative acoustic activity detected. "
            f"The identified source is **"
            f"{predicted_class.replace('_', ' ').title()}**, "
            f"belonging to the **{category}** category. "
            f"Additional recordings can help determine "
            f"whether the source is persistent."
        )
    else:
        return (
            f"Low relative acoustic activity detected. "
            f"The identified source is **"
            f"{predicted_class.replace('_', ' ').title()}**, "
            f"belonging to the **{category}** category."
        )
# =========================================================
# PAGE HEADER
# =========================================================
st.title("🎧 NoiseGuard")
st.subheader(
    "AI-Based Environmental Noise Monitoring & Source Analysis"
)
st.write(
    "Upload environmental audio recordings and NoiseGuard "
    "will identify potential noise sources, analyze their "
    "relative acoustic activity, and maintain a monitoring "
    "history during the current session."
)
st.info(
    "NoiseGuard performs environmental sound-source "
    "classification and relative acoustic activity analysis. "
    "It does not directly measure calibrated sound pressure "
    "level (dB)."
)
# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🎧 NoiseGuard")
st.sidebar.write(
    "Environmental Noise Monitoring System"
)
st.sidebar.divider()
st.sidebar.markdown(
    """
### Monitoring Pipeline
🎙️ Audio Input
↓
🔧 Feature Extraction
↓
🤖 SVM Classification
↓
🔊 Noise Source Detection
↓
📊 Acoustic Activity Analysis
↓
📈 Monitoring History
↓
🌍 Environmental Assessment
"""
)
st.sidebar.divider()
st.sidebar.caption(
    "Relative acoustic activity is calculated from "
    "the uploaded recording and is not a calibrated "
    "noise measurement."
)
# =========================================================
# AUDIO UPLOAD
# =========================================================
st.header("🎙️ Environmental Audio Monitoring")
st.write("Try a built-in sample to explore NoiseGuard, or upload your own recording.")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_audio")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".flac")
sample_files = sorted(
    name for name in os.listdir(SAMPLE_DIR)
    if name.lower().endswith(AUDIO_EXTENSIONS)
) if os.path.isdir(SAMPLE_DIR) else []
input_mode = st.radio(
    "Choose audio source",
    ["Try a sample", "Upload audio"],
    horizontal=True,
)
uploaded_file = None
if input_mode == "Try a sample":
    if sample_files:
        selected_sample = st.selectbox("Choose a sample recording", sample_files)
        sample_path = os.path.join(SAMPLE_DIR, selected_sample)
        with open(sample_path, "rb") as sample_handle:
            uploaded_file = io.BytesIO(sample_handle.read())
        uploaded_file.name = selected_sample
    else:
        st.warning(
            "No built-in audio samples were found. Create a folder named "
            "'sample_audio' beside app.py and place a few .wav, .mp3, .ogg, "
            "or .flac sample clips inside it. The clips will then be available "
            "to anyone running this app."
        )
else:
    uploaded_file = st.file_uploader(
        "Upload an environmental audio recording",
        type=["wav", "mp3", "ogg", "flac"],
    )
if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button(
        "🔍 Analyze & Monitor Audio",
        type="primary"
    ):
        with st.spinner(
            "Analyzing environmental audio..."
        ):
            suffix = os.path.splitext(
                uploaded_file.name
            )[1]
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(
                    uploaded_file.getbuffer()
                )
                temp_path = temp_file.name
            try:
                # ---------------------------------------------
                # AI CLASSIFICATION
                # ---------------------------------------------
                features = extract_rich_features(
                    temp_path
                )
                features = features.reshape(
                    1,
                    -1
                )
                prediction = model.predict(
                    features
                )[0]
                predicted_class = (
                    label_encoder
                    .inverse_transform(
                        [prediction]
                    )[0]
                )
                probabilities = (
                    model.predict_proba(
                        features
                    )[0]
                )
                confidence = (
                    probabilities[prediction] * 100
                )
                category = sound_categories.get(
                    predicted_class,
                    "Environmental Sound"
                )
                # ---------------------------------------------
                # ACOUSTIC ANALYSIS
                # ---------------------------------------------
                acoustic = analyze_acoustic_activity(
                    temp_path
                )
                activity = acoustic["activity"]
                # ---------------------------------------------
                # MONITORING EVENT
                # ---------------------------------------------
                monitoring_event = {
                    "Time": datetime.now().strftime(
                        "%H:%M:%S"
                    ),
                    "Source": predicted_class.replace(
                        "_",
                        " "
                    ).title(),
                    "Category": category,
                    "Confidence (%)": round(
                        confidence,
                        2
                    ),
                    "Activity": activity,
                    "Active Portion (%)": acoustic[
                        "active_percentage"
                    ],
                    "Duration (s)": acoustic[
                        "duration"
                    ]
                }
                st.session_state.monitoring_history.append(
                    monitoring_event
                )
                # ---------------------------------------------
                # SUCCESS
                # ---------------------------------------------
                st.success(
                    "Environmental audio monitoring completed!"
                )
                # ---------------------------------------------
                # AI IDENTIFICATION
                # ---------------------------------------------
                st.subheader(
                    "🤖 AI Sound Identification"
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Detected Sound",
                        predicted_class
                        .replace(
                            "_",
                            " "
                        )
                        .title()
                    )
                with col2:
                    st.metric(
                        "AI Confidence",
                        f"{confidence:.2f}%"
                    )
                with col3:
                    st.metric(
                        "Environmental Category",
                        category
                    )
                # ---------------------------------------------
                # ACOUSTIC MONITORING
                # ---------------------------------------------
                st.subheader(
                    "🌍 Acoustic Monitoring"
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Recording Duration",
                        f"{acoustic['duration']:.1f} s"
                    )
                with col2:
                    st.metric(
                        "Relative Acoustic Activity",
                        activity
                    )
                with col3:
                    st.metric(
                        "Active Acoustic Portion",
                        f"{acoustic['active_percentage']:.2f}%"
                    )
                st.caption(
                    "Acoustic activity is calculated relative "
                    "to the uploaded recording. It is not a "
                    "calibrated sound-pressure measurement."
                )
                # ---------------------------------------------
                # ENVIRONMENTAL ASSESSMENT
                # ---------------------------------------------
                st.subheader(
                    "🌱 Environmental Assessment"
                )
                assessment = generate_assessment(
                    predicted_class,
                    category,
                    activity
                )
                if activity == "High":
                    st.warning(
                        assessment
                    )
                elif activity == "Moderate":
                    st.info(
                        assessment
                    )
                else:
                    st.success(
                        assessment
                    )
                # ---------------------------------------------
                # TOP PREDICTIONS
                # ---------------------------------------------
                st.subheader(
                    "🔎 Top Sound Predictions"
                )
                top_indices = np.argsort(
                    probabilities
                )[::-1][:3]
                for idx in top_indices:
                    class_name = (
                        label_encoder
                        .inverse_transform(
                            [idx]
                        )[0]
                    )
                    probability = (
                        probabilities[idx] * 100
                    )
                    st.write(
                        f"**{class_name.replace('_', ' ').title()}** "
                        f"— {probability:.2f}%"
                    )
                    st.progress(
                        float(
                            probability / 100
                        )
                    )
                # ---------------------------------------------
                # SPECTROGRAM
                # ---------------------------------------------
                st.subheader(
                    "📊 Audio Spectrogram"
                )
                audio, sr = librosa.load(
                    temp_path,
                    sr=None
                )
                spectrogram = (
                    librosa.feature.melspectrogram(
                        y=audio,
                        sr=sr,
                        n_mels=128
                    )
                )
                spectrogram_db = (
                    librosa.power_to_db(
                        spectrogram,
                        ref=np.max
                    )
                )
                fig, ax = plt.subplots(
                    figsize=(12, 4)
                )
                librosa.display.specshow(
                    spectrogram_db,
                    sr=sr,
                    x_axis="time",
                    y_axis="mel",
                    ax=ax
                )
                ax.set_title(
                    "NoiseGuard Audio Spectrogram"
                )
                ax.set_xlabel(
                    "Time"
                )
                ax.set_ylabel(
                    "Frequency"
                )
                plt.tight_layout()
                st.pyplot(
                    fig
                )
                plt.close(
                    fig
                )
                # ---------------------------------------------
                # CURRENT RECORDING INTERPRETATION
                # ---------------------------------------------
                st.subheader(
                    "📋 Environmental Interpretation"
                )
                st.write(
                    f"The detected sound is classified as "
                    f"**{predicted_class.replace('_', ' ').title()}**, "
                    f"which belongs to the **{category}** category."
                )
                st.write(
                    f"The recording contains "
                    f"**{acoustic['active_percentage']:.2f}%** "
                    f"relative acoustic activity and is classified "
                    f"as **{activity} acoustic activity**."
                )
                st.write(
                    "This monitoring sample can contribute to "
                    "time-based environmental noise analysis "
                    "when combined with additional recordings."
                )
            finally:
                if os.path.exists(
                    temp_path
                ):
                    os.remove(
                        temp_path
                    )
# =========================================================
# MONITORING DASHBOARD
# =========================================================
st.divider()
st.header(
    "📡 Noise Pollution Monitoring Dashboard"
)
if len(
    st.session_state.monitoring_history
) == 0:
    st.info(
        "No monitoring samples recorded yet. "
        "Upload and analyze audio recordings above "
        "to build the monitoring history."
    )
else:
    history_df = pd.DataFrame(
        st.session_state.monitoring_history
    )
    # ---------------------------------------------
    # SUMMARY METRICS
    # ---------------------------------------------
    total_recordings = len(
        history_df
    )
    total_duration = history_df[
        "Duration (s)"
    ].sum()
    high_events = (
        history_df["Activity"] == "High"
    ).sum()
    most_common_source = (
        history_df["Source"]
        .value_counts()
        .idxmax()
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Monitoring Samples",
            total_recordings
        )
    with col2:
        st.metric(
            "Total Duration",
            f"{total_duration:.1f} s"
        )
    with col3:
        st.metric(
            "High-Activity Events",
            int(high_events)
        )
    with col4:
        st.metric(
            "Most Frequent Source",
            most_common_source
        )
    # ---------------------------------------------
    # MONITORING HISTORY TABLE
    # ---------------------------------------------
    st.subheader(
        "🕒 Monitoring History"
    )
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )
    # ---------------------------------------------
    # SOURCE DISTRIBUTION
    # ---------------------------------------------
    st.subheader(
        "🔊 Detected Noise Source Distribution"
    )
    source_counts = (
        history_df["Source"]
        .value_counts()
    )
    st.bar_chart(
        source_counts
    )
    # ---------------------------------------------
    # ACTIVITY TREND
    # ---------------------------------------------
    st.subheader(
        "📈 Relative Acoustic Activity Trend"
    )
    activity_mapping = {
        "Low": 1,
        "Moderate": 2,
        "High": 3
    }
    trend_df = history_df.copy()
    trend_df["Activity Score"] = (
        trend_df["Activity"]
        .map(activity_mapping)
    )
    trend_df["Monitoring Sample"] = range(
        1,
        len(trend_df) + 1
    )
    st.line_chart(
        trend_df.set_index(
            "Monitoring Sample"
        )["Activity Score"]
    )
    st.caption(
        "Activity scores are relative: "
        "1 = Low, 2 = Moderate, 3 = High. "
        "They do not represent calibrated dB values."
    )
    # ---------------------------------------------
    # MONITORING SUMMARY
    # ---------------------------------------------
    st.subheader(
        "🌍 Monitoring Summary"
    )
    average_active = history_df[
        "Active Portion (%)"
    ].mean()
    st.write(
        f"NoiseGuard analyzed **{total_recordings} "
        f"monitoring sample(s)** covering approximately "
        f"**{total_duration:.1f} seconds** of audio."
    )
    st.write(
        f"The average relative active acoustic portion "
        f"across the monitoring samples was "
        f"**{average_active:.2f}%**."
    )
    st.write(
        f"The most frequently detected source was "
        f"**{most_common_source}**."
    )
    if high_events > 0:
        st.warning(
            f"{high_events} high relative acoustic activity "
            f"event(s) were detected. Additional monitoring "
            f"can help determine whether these events are "
            f"persistent."
        )
    else:
        st.success(
            "No high relative acoustic activity events "
            "have been recorded in the current monitoring session."
        )
# =========================================================
