import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix

# ===============================
# KONFIGURASI
# ===============================
st.set_page_config(page_title="Deteksi Tekstur (Matriks)", layout="centered")
st.title("📊 Deteksi Tekstur Citra Berbasis Matriks")

# ===============================
# FUNGSI DASAR
# ===============================
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.uint8)

# ===============================
# LBP MATRIX (Histogram)
# ===============================
def lbp_matrix(gray):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    df = pd.DataFrame(hist.reshape(1, -1))
    df.columns = [f"P{i}" for i in range(df.shape[1])]
    return df

# ===============================
# GLCM MATRIX
# ===============================
def glcm_matrix(gray):
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=8,   # DIPERKECIL supaya tabel masuk akal
        symmetric=True,
        normed=True
    )

    matrix = glcm[:, :, 0, 0]
    df = pd.DataFrame(matrix)
    return df

# ===============================
# FOURIER MATRIX
# ===============================
def fft_matrix(gray, size=10):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2

    cropped = magnitude[
        center_h-size//2:center_h+size//2,
        center_w-size//2:center_w+size//2
    ]

    df = pd.DataFrame(cropped)
    return df

# ===============================
# INPUT GAMBAR
# ===============================
uploaded_file = st.file_uploader(
    "📂 Upload citra (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = load_image(uploaded_file)
    gray = to_gray(img)

    # ===== LBP =====
    st.subheader("🔹 Matriks Histogram LBP")
    st.caption("Representasi frekuensi pola LBP (vektor / matriks)")
    st.dataframe(lbp_matrix(gray), use_container_width=True)

    # ===== GLCM =====
    st.subheader("🔹 Matriks GLCM")
    st.caption("Matriks co-occurrence (levels=8, distance=1, angle=0°)")
    st.dataframe(glcm_matrix(gray), use_container_width=True)

    # ===== FFT =====
    st.subheader("🔹 Matriks Fourier Transform")
    st.caption("Magnitude spectrum (dipotong area tengah)")
    st.dataframe(fft_matrix(gray), use_container_width=True)

else:
    st.info("Silakan upload citra terlebih dahulu.")
