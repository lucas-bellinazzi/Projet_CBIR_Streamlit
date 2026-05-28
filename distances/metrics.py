import numpy as np
from scipy.spatial.distance import euclidean, cityblock, chebyshev, canberra

def get_similar_images(query, base, method="euclidean", top_k=5):
  dists = []
  for idx, feat in enumerate(base):
    if method == "euclidean":
      dist = euclidean(query, feat)
    elif method == "manhattan":
      dist = cityblock(query, feat)
    elif method == "chebyshev":
      dist = chebyshev(query, feat)
    elif method == "canberra":
      dist = canberra(query, feat)
    else:
      raise ValueError("Unsupported method")
    dists.append((idx, dist))
  return sorted(dists, key=lambda x: x[1])[:top_k]
