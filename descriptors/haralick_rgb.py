import numpy as np
from mahotas.features import haralick

def extract_features(image):
  features = []

  for i in range(3):  # R, G, B
    channel = image[:, :, i]
    feats = haralick(channel).mean(0).tolist()
    feats = [float(x) for x in feats]
    features.extend(feats)

  return features
