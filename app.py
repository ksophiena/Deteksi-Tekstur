import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# ===============================
# KONFIGURASI STREAMLIT
# ===============================
st.set_page_config(page_title="Ekstraksi Tekstur", layout="centered")
st.title("📊 Tabel Hasil Deteksi Tekstur Citra")

st.write("Upload citra, lalu sistem akan menampilkan **tabel fitur tekstur**.")

# ===============================
# FUNGSI
# ===============================
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)

def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.uint8)  # PENTING

# ===============================
# LBP FEATURES
# ===============================
def lbp_features(gray):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    return {
        "LBP_Mean": float(np.mean(lbp)),
        "LBP_Variance": float(np.var(lbp)),
        "LBP_StdDev": float(np.std(lbp))
    }

# ===============================
# GLCM FEATURES
# ===============================
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
        "GLCM_Contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "GLCM_Dissimilarity": float(graycoprops(glcm, "dissimilarity")[0, 0]),
        "GLCM_Homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "GLCM_Energy": float(graycoprops(glcm, "energy")[0, 0]),
        "GLCM_Correlation": float(graycoprops(glcm, "correlation")[0, 0]),
        "GLCM_ASM": float(graycoprops(glcm, "ASM")[0, 0])
    }

# ===============================
# FOURIER FEATURES
# ===============================
def fourier_features(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)  # AMAN

    return {
        "FFT_Mean": float(np.mean(magnitude)),
        "FFT_Variance": float(np.var(magnitude)),
        "FFT_Energy": float(np.sum(magnitude ** 2))
    }

# ===============================
# INPUT GAMBAR
# ===============================
uploaded_file = st.file_uploader(
    "📂 Upload citra (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        img = load_image(uploaded_file)
        gray = to_gray(img)

        features = {}
        features.update(lbp_features(gray))
        features.update(glcm_features(gray))
        features.update(fourier_features(gray))

        df = pd.DataFrame(
            list(features.items()),
            columns=["Fitur Tekstur", "Nilai"]
        )

        st.subheader("📌 Hasil Ekstraksi Tekstur")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            csv,
            "hasil_tekstur.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Terjadi error: {e}")

else:
    st.info("Silakan upload citra terlebih dahulu.")
