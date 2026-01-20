import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# -------------------------------
# KONFIGURASI
# -------------------------------
st.set_page_config(page_title="Ekstraksi Tekstur", layout="centered")
st.title("📊 Tabel Hasil Deteksi Tekstur Citra")

# -------------------------------
# FUNGSI
# -------------------------------
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# -------------------------------
# LBP FEATURES
# -------------------------------
def lbp_features(gray):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    return {
        "LBP_Mean": np.mean(lbp),
        "LBP_Variance": np.var(lbp),
        "LBP_StdDev": np.std(lbp)
    }

# -------------------------------
# GLCM FEATURES
# -------------------------------
def glcm_features(gray):
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    return {
        "GLCM_Contrast": graycoprops(glcm, "contrast")[0, 0],
        "GLCM_Dissimilarity": graycoprops(glcm, "dissimilarity")[0, 0],
        "GLCM_Homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
        "GLCM_Energy": graycoprops(glcm, "energy")[0, 0],
        "GLCM_Correlation": graycoprops(glcm, "correlation")[0, 0],
        "GLCM_ASM": graycoprops(glcm, "ASM")[0, 0]
    }

# -------------------------------
# FOURIER FEATURES
# -------------------------------
def fourier_features(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    return {
        "FFT_Mean": np.mean(magnitude),
        "FFT_Variance": np.var(magnitude),
        "FFT_Energy": np.sum(magnitude ** 2)
    }

# -------------------------------
# UPLOAD IMAGE
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload citra (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = load_image(uploaded_file)
    gray = to_gray(img)

    # Ekstraksi fitur
    features = {}
    features.update(lbp_features(gray))
    features.update(glcm_features(gray))
    features.update(fourier_features(gray))

    # Konversi ke tabel
    df = pd.DataFrame(features.items(), columns=["Fitur", "Nilai"])

    st.subheader("📌 Tabel Hasil Deteksi Tekstur")
    st.dataframe(df, use_container_width=True)

    # Optional: download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        csv,
        "hasil_tekstur.csv",
        "text/csv"
    )
else:
    st.info("Silakan upload citra untuk melihat tabel hasil ekstraksi tekstur.")
