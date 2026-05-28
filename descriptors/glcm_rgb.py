import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_features(image):
  features = []

  for i in range(3):
    channel = image[:,:,i]
    co_matrix = graycomatrix(channel, distances=[1], angles=[3*np.pi/2], symmetric=False, normed=True)
    contrast = graycoprops(co_matrix,'contrast')[0,0]
    dissimilarity = graycoprops(co_matrix,'dissimilarity')[0,0]
    homogeneity = graycoprops(co_matrix,'homogeneity')[0,0]
    correlation = graycoprops(co_matrix,'correlation')[0,0]
    energy = graycoprops(co_matrix,'energy')[0,0]
    ASM = graycoprops(co_matrix,'ASM')[0,0]
    feats = [contrast,dissimilarity,homogeneity,correlation,energy,ASM]
    feats = [ float(x) for x in feats ]

    features.extend(feats)

  return features
