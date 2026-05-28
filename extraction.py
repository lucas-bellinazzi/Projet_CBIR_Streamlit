import os
import numpy as np
from PIL import Image
from descripteurs import glcm_rgb, haralick_rgb, bit_rgb, fusion
from utils.image_loader import load_and_preprocess_image

dict_class = {
  'iris-setosa': 0,
  'iris-versicolour': 1,
  'iris-virginica': 2
}

def extraction_signatures(chemin_repertoire):
  
  list_glcm = []
  list_haralick = []
  list_bit = []
  list_fusion = []

  for root, _, files in os.walk(chemin_repertoire):
    for file in files:
      if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        chemin = os.path.join(root, file)
        relative_path = os.path.relpath(chemin, chemin_repertoire)

        img_pil = Image.open(chemin).convert("RGB")
        img_np = load_and_preprocess_image(img_pil)

        list_glcm.append(glcm_rgb.extract_features(img_np))
        list_haralick.append(haralick_rgb.extract_features(img_np))
        list_bit.append(bit_rgb.extract_features(img_np))
        list_fusion.append(fusion.extract_features(img_np))

        print(f"Signatures extraites pour: {relative_path}")

  print("\nVérification des dimensions :")
  print(f"GLCM: {len(list_glcm[0]) if list_glcm else 0} features")
  print(f"Haralick: {len(list_haralick[0]) if list_haralick else 0} features")
  print(f"BiT: {len(list_bit[0]) if list_bit else 0} features")
  print(f"Fusion: {len(list_fusion[0]) if list_fusion else 0} features")

  np.save("signatures/GLCM_RGB.npy", list_glcm)
  np.save("signatures/Haralick_RGB.npy", list_haralick)
  np.save("signatures/BiT_RGB.npy", list_bit)
  np.save("signatures/Fusion.npy", list_fusion)

  print("\nSignatures et métadonnées enregistrées avec succès !")

if __name__ == "__main__":
  extraction_signatures("./data/dataset/")