
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import tempfile
import os


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="NoiseGuard",
    page_icon="🎧",
    layout="wide"
)


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

MODEL_PATH = "noiseguard_final_model.pkl"
ENCODER_PATH = "noiseguard_final_label_encoder.pkl"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


# ---------------------------------------------------------
# Environmental categories
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Feature extraction
# Must match training exactly
# ---------------------------------------------------------

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

    # Same 112-feature representation used during training
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


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

st.title("🎧 NoiseGuard")

st.subheader(
    "AI-Based Environmental Noise Source Analyzer"
)

st.write(
    "Upload an environmental audio clip and NoiseGuard "
    "will identify the most likely sound source using "
    "machine learning."
)

st.info(
    "NoiseGuard performs environmental sound-source "
    "classification. It does not directly measure "
    "sound pressure level (dB)."
)


# ---------------------------------------------------------
# Upload audio
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Environmental Audio",
    type=["wav", "mp3", "ogg", "flac"]
)


if uploaded_file is not None:

    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    if st.button(
        "🔍 Analyze Audio",
        type="primary"
    ):

        with st.spinner("Analyzing audio..."):

            # Save uploaded file temporarily
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

                # Extract features
                features = extract_rich_features(
                    temp_path
                )

                features = features.reshape(
                    1, -1
                )

                # Prediction
                prediction = model.predict(
                    features
                )[0]

                predicted_class = (
                    label_encoder
                    .inverse_transform([prediction])[0]
                )

                probabilities = (
                    model.predict_proba(features)[0]
                )

                confidence = (
                    probabilities[prediction] * 100
                )

                category = sound_categories.get(
                    predicted_class,
                    "Environmental Sound"
                )

                # -------------------------------------------------
                # Results
                # -------------------------------------------------

                st.success(
                    "Audio analysis completed!"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Detected Sound",
                        predicted_class
                        .replace("_", " ")
                        .title()
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                with col3:
                    st.metric(
                        "Environmental Category",
                        category
                    )

                # -------------------------------------------------
                # Top 3 predictions
                # -------------------------------------------------

                st.subheader(
                    "Top Sound Predictions"
                )

                top_indices = np.argsort(
                    probabilities
                )[::-1][:3]

                for idx in top_indices:

                    class_name = (
                        label_encoder
                        .inverse_transform([idx])[0]
                    )

                    probability = (
                        probabilities[idx] * 100
                    )

                    st.write(
                        f"**{class_name.replace('_', ' ').title()}** "
                        f"— {probability:.2f}%"
                    )

                    st.progress(
                        float(probability / 100)
                    )

                # -------------------------------------------------
                # Spectrogram
                # -------------------------------------------------

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

                ax.set_xlabel("Time")
                ax.set_ylabel("Frequency")

                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

                # -------------------------------------------------
                # Interpretation
                # -------------------------------------------------

                st.subheader(
                    "🌍 Environmental Interpretation"
                )

                st.write(
                    f"The detected sound is classified as "
                    f"**{predicted_class.replace('_', ' ').title()}**, "
                    f"which belongs to the **{category}** category."
                )

            finally:

                # Remove temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)


# ---------------------------------------------------------
# Project information
# ---------------------------------------------------------

st.divider()

st.subheader("About NoiseGuard")

st.write(
    "NoiseGuard uses machine learning and audio signal "
    "processing to classify environmental sound sources. "
    "The system uses MFCC, chroma, spectral and "
    "zero-crossing features with an SVM classifier."
)

st.caption(
    "Final model test accuracy: 75.03% on 837 held-out "
    "UrbanSound8K test clips."
)
