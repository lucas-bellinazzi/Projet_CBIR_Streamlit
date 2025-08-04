import numpy as np
import cv2

def load_and_preprocess_image(image, size=(128, 128)):
  # imagem PIL → np.array RGB
  image = np.array(image.resize(size))
  if image.shape[2] == 4:
    image = image[:, :, :3]  # remove canal alpha
  return image
