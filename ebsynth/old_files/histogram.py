import cv2
import numpy as np
import matplotlib.pyplot as plt

# HistogramMatcher class definition (same as before)
class HistogramMatcher:
    def __init__(self):
        pass

    def match(self, Oai, Obi):
        Oai = cv2.cvtColor(Oai, cv2.COLOR_BGR2Lab)
        Obi = cv2.cvtColor(Obi, cv2.COLOR_BGR2Lab)
        Oai_mean, Oai_std = self.computeMeanAndStd(Oai)
        Obi_mean, Obi_std = self.computeMeanAndStd(Obi)
        t_mean = (Oai_mean + Obi_mean) / 2.0
        t_std = (Oai_std + Obi_std) / 2.0
        Oai_transformed = self.histogramTransform(Oai, Oai_mean, Oai_std, t_mean, t_std)
        Obi_transformed = self.histogramTransform(Obi, Obi_mean, Obi_std, t_mean, t_std)
        Oai_transformed = cv2.cvtColor(Oai_transformed.astype(np.uint8), cv2.COLOR_Lab2BGR)
        Obi_transformed = cv2.cvtColor(Obi_transformed.astype(np.uint8), cv2.COLOR_Lab2BGR)
        return Oai_transformed, Obi_transformed

    def computeMeanAndStd(self, img):
        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))
        return mean, std

    def histogramTransform(self, img, inputMean, inputStd, targetMean, targetStd):
        transformed_img = (img - inputMean) * (targetStd / inputStd) + targetMean
        np.clip(transformed_img, 0, 255, out=transformed_img)
        return transformed_img

# Create the HistogramMatcher object
#matcher = HistogramMatcher()

# Perform histogram matching
#Oai_transformed, Obi_transformed = matcher.match(Oai, Obi)
