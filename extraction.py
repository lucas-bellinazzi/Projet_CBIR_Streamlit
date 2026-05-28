import os
import numpy as np
from PIL import Image
from descriptors import glcm_rgb, haralick_rgb, bit_rgb, fusion
from utils.image_loader import load_and_preprocess_image

dict_class = {
  'iris-setosa': 0,
  'iris-versicolour': 1,
  'iris-virginica': 2
}

def extract_signatures(directory_path):
  
  glcm_list = []
  haralick_list = []
  bit_list = []
  fusion_list = []

  for root, _, files in os.walk(directory_path):
    for file in files:
      if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        file_path = os.path.join(root, file)
        relative_path = os.path.relpath(file_path, directory_path)

        img_pil = Image.open(file_path).convert("RGB")
        img_np = load_and_preprocess_image(img_pil)

        glcm_list.append(glcm_rgb.extract_features(img_np))
        haralick_list.append(haralick_rgb.extract_features(img_np))
        bit_list.append(bit_rgb.extract_features(img_np))
        fusion_list.append(fusion.extract_features(img_np))

        print(f"Features extracted for: {relative_path}")

  print("\nDimension verification:")
  print(f"GLCM: {len(glcm_list[0]) if glcm_list else 0} features")
  print(f"Haralick: {len(haralick_list[0]) if haralick_list else 0} features")
  print(f"BiT: {len(bit_list[0]) if bit_list else 0} features")
  print(f"Fusion: {len(fusion_list[0]) if fusion_list else 0} features")

  np.save("signatures/GLCM_RGB.npy", glcm_list)
  np.save("signatures/Haralick_RGB.npy", haralick_list)
  np.save("signatures/BiT_RGB.npy", bit_list)
  np.save("signatures/Fusion.npy", fusion_list)

  print("\nFeatures and metadata successfully saved!")

if __name__ == "__main__":
  extract_signatures("./data/dataset/")