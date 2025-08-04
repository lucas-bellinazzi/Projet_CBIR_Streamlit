from descripteurs import glcm_rgb, haralick_rgb, bit_rgb
import numpy as np

def extract_features(image):
  f1 = glcm_rgb.extract_features(image)
  f2 = haralick_rgb.extract_features(image)
  f3 = bit_rgb.extract_features(image)
  
  return np.concatenate([f1, f2, f3])
