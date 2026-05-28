import streamlit as st
import numpy as np
from PIL import Image
from utils.image_loader import load_and_preprocess_image
from distances.metrics import get_similar_images
from descripteurs import glcm_rgb, haralick_rgb, bit_rgb, fusion

st.title("Recherche d'Images Basée sur le Contenu (CBIR)")

# -- Upload
uploaded_file = st.file_uploader("Téléversez une image", type=["jpg", "png", "jpeg"])

# -- Options
nb_images = st.slider("Nombre d'images similaires à afficher", 1, 10, 5)
descripteur = st.selectbox("Descripteur à utiliser", ["GLCM", "Haralick", "BiT", "Fusion"])
distance = st.selectbox("Mesure de distance", ["Euclidienne", "Manhattan", "Tchebychev", "Canberra"])

# -- Recherche
if uploaded_file:
  img = Image.open(uploaded_file).convert("RGB")
  st.image(img, caption="Image recherchée", width="stretch")

  with st.spinner("Recherche en cours..."):
    # Prétraitement et extraction
    query = load_and_preprocess_image(img)

    if descripteur == "GLCM":
      vector = glcm_rgb.extract_features(query)
      base = np.load("Signatures/GLCM_RGB.npy", allow_pickle=True)
    elif descripteur == "Haralick":
      vector = haralick_rgb.extract_features(query)
      base = np.load("Signatures/Haralick_RGB.npy", allow_pickle=True)
    elif descripteur == "BiT":
      vector = bit_rgb.extract_features(query)
      base = np.load("Signatures/BiT_RGB.npy", allow_pickle=True)
    else:
      vector = fusion.extract_features(query)
      base = np.load("Signatures/Fusion.npy", allow_pickle=True)

    vector_np = np.array(vector)
    base_np = np.array([x[:-2] for x in base])

    if vector_np.shape[0] != base_np.shape[1]:
      st.error(f"""
        ERREUR: Dimensions incohérentes
        Descripteur: {descripteur}
        Requête: {vector_np.shape[0]} features
        Base: {base_np.shape[1]} features
      """)
      min_dim = min(vector_np.shape[0], base_np.shape[1])
      vector_np = vector_np[:min_dim]
      base_np = base_np[:, :min_dim]
      st.warning(f"Utilisation des premières {min_dim} features seulement")

    # Recherche des plus proches
    results = get_similar_images(vector_np, base_np, method=distance.lower(), top_k=nb_images)

  # -- Affichage des résultats
  st.subheader("Résultats Similaires")
  cols = st.columns(nb_images)
  for i, (idx, score) in enumerate(results):
    img_path = f"data/dataset/img_{idx}.jpg"
    with cols[i]:
      st.image(img_path, caption=f"Score: {score:.2f}", use_container_width=True)
