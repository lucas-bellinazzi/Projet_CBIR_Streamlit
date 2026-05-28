from descriptors import glcm_rgb, haralick_rgb, bit_rgb
import numpy as np

def extract_features(image):
  f1 = glcm_rgb.extract_features(image)
  f2 = haralick_rgb.extract_features(image)
  f3 = bit_rgb.extract_features(image)
  
  if not isinstance(f1, list): f1 = f1.tolist()
  if not isinstance(f2, list): f2 = f2.tolist()
  if not isinstance(f3, list): f3 = f3.tolist()

  return f1 + f2 + f3
