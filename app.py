import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix

# ======================================
# KONFIGURASI
# ======================================
st.set_page_config(
    page_title="Deteksi Tekstur Citra",
    layout="centered"
)

st.title("Deteksi Tekstur Citra (LBP & GLCM)")

# ======================================
# FUNGSI DASAR
# ======================================
def load_image(uploaded_file):
    return np.array(Image.open(uploaded_file).convert("RGB"))

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def show_matrix(df):
    html = df.to_html(index=False, header=False)
    st.markdown(html, unsafe_allow_html=True)

# ======================================
# LBP
# ======================================
def lbp_matrix(gray):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="default")
    patch = lbp[:5, :5]
    return pd.DataFrame(patch)

# ======================================
# GLCM
# ======================================
def glcm_matrix(gray):
    gray_q = (gray / 64).astype(np.uint8)
    glcm = graycomatrix(
        gray_q,
        distances=[1],
        angles=[0],
        levels=4,
        symmetric=True,
        normed=True
    )
    return pd.DataFrame(glcm[:, :, 0, 0])

# ======================================
# INPUT GAMBAR
# ======================================
uploaded_file = st.file_uploader(
    "Upload citra (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = load_image(uploaded_file)
    gray = to_gray(img)

    # ===============================
    # CITRA ASLI
    # ===============================
    st.subheader("Citra Asli")
    st.image(img, use_container_width=True)

    # ===============================
    # LBP
    # ===============================
    st.subheader("Local Binary Pattern (LBP)")
    show_matrix(lbp_matrix(gray))

    # ===============================
    # GLCM
    # ===============================
    st.subheader("Gray Level Co-occurrence Matrix (GLCM)")
    show_matrix(glcm_matrix(gray))

else:
    st.info("Silakan upload citra untuk memulai.")
