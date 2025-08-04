import os
import cv2
import numpy as np
from descripteurs import glcm_rgb, haralick_rgb, bit_rgb, fusion
from utils.image_loader import load_and_preprocess_image
from PIL import Image

def extraction_signatures(chemin_repertoire, taille=(128, 128)):
  list_glcm = []
  list_haralick = []
  list_bit = []
  list_fusion = []

  for root, _, files in os.walk(chemin_repertoire):
    for file in files:
      if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        chemin = os.path.join(root, file)
        relative_path = os.path.relpath(chemin, chemin_repertoire)
        class_name = os.path.dirname(relative_path)

        img_pil = Image.open(chemin).convert("RGB")
        img_np = load_and_preprocess_image(img_pil, size=taille)

        glcm_feat = glcm_rgb.extract_features(img_np) + [class_name, relative_path]
        haralick_feat = haralick_rgb.extract_features(img_np) + [class_name, relative_path]
        bit_feat = bit_rgb.extract_features(img_np) + [class_name, relative_path]
        fusion_feat = fusion.extract_features(img_np).tolist() + [class_name, relative_path]

        list_glcm.append(glcm_feat)
        list_haralick.append(haralick_feat)
        list_bit.append(bit_feat)
        list_fusion.append(fusion_feat)

        print(f"Signatures extraites pour: {relative_path}")

  np.save("signatures/GLCM_RGB.npy", np.array(list_glcm, dtype=object))
  np.save("signatures/Haralick_RGB.npy", np.array(list_haralick, dtype=object))
  np.save("signatures/BiT_RGB.npy", np.array(list_bit, dtype=object))
  np.save("signatures/Fusion.npy", np.array(list_fusion, dtype=object))

  print("Signatures enregistrées avec succès !")

if __name__ == "__main__":
  extraction_signatures("data/dataset")
