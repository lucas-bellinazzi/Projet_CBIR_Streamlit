import numpy as np
from scipy.spatial.distance import euclidean, cityblock, chebyshev, canberra

def get_similar_images(query, base, method="euclidienne", top_k=5):
  dists = []
  for idx, feat in enumerate(base):
    if method == "euclidienne":
      dist = euclidean(query, feat)
    elif method == "manhattan":
      dist = cityblock(query, feat)
    elif method == "tchebychev":
      dist = chebyshev(query, feat)
    elif method == "canberra":
      dist = canberra(query, feat)
    else:
      raise ValueError("Méthode non supportée")
    dists.append((idx, dist))
  return sorted(dists, key=lambda x: x[1])[:top_k]
