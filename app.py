import streamlit as st
import numpy as np
from PIL import Image
from utils.image_loader import load_and_preprocess_image
from distances.metrics import get_similar_images
from descriptors import glcm_rgb, haralick_rgb, bit_rgb, fusion

st.title("Content-Based Image Retrieval (CBIR)")

# -- Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# -- Options
num_images = st.slider("Number of similar images to display", 1, 10, 5)
descriptor = st.selectbox("Descriptor to use", ["GLCM", "Haralick", "BiT", "Fusion"])
distance = st.selectbox("Distance metric", ["Euclidean", "Manhattan", "Chebyshev", "Canberra"])

# -- Search
if uploaded_file:
  img = Image.open(uploaded_file).convert("RGB")
  st.image(img, caption="Query image", width="stretch")

  with st.spinner("Searching..."):
    # Preprocessing and extraction
    query = load_and_preprocess_image(img)

    if descriptor == "GLCM":
      vector = glcm_rgb.extract_features(query)
      base = np.load("signatures/GLCM_RGB.npy", allow_pickle=True)
    elif descriptor == "Haralick":
      vector = haralick_rgb.extract_features(query)
      base = np.load("signatures/Haralick_RGB.npy", allow_pickle=True)
    elif descriptor == "BiT":
      vector = bit_rgb.extract_features(query)
      base = np.load("signatures/BiT_RGB.npy", allow_pickle=True)
    else:
      vector = fusion.extract_features(query)
      base = np.load("signatures/Fusion.npy", allow_pickle=True)

    vector_np = np.array(vector)
    base_np = np.array([x[:-2] for x in base])

    if vector_np.shape[0] != base_np.shape[1]:
      st.error(f"""
        ERROR: Inconsistent dimensions
        Descriptor: {descriptor}
        Query: {vector_np.shape[0]} features
        Database: {base_np.shape[1]} features
      """)
      min_dim = min(vector_np.shape[0], base_np.shape[1])
      vector_np = vector_np[:min_dim]
      base_np = base_np[:, :min_dim]
      st.warning(f"Using only the first {min_dim} features")

    # Search for nearest neighbors
    results = get_similar_images(vector_np, base_np, method=distance.lower(), top_k=num_images)

  # -- Display results
  st.subheader("Similar Results")
  cols = st.columns(num_images)
  for i, (idx, score) in enumerate(results):
    img_path = f"data/dataset/img_{idx}.jpg"
    with cols[i]:
      st.image(img_path, caption=f"Score: {score:.2f}", use_container_width=True)
