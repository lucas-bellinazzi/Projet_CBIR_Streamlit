from BiT import bio_taxo

def extract_features(image):
  features = []

  for i in range(3):  # R, G, B
    channel = image[:, :, i]
    feats = bio_taxo(channel)
    feats=[float(x) for x in feats]
    features.extend(feats)

  return features
