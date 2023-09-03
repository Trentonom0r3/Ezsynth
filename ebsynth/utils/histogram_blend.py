import cv2
import numpy as np

class HistogramBlender:
    def __init__(self):
        pass

    def blend(self, Oai, Obi, errMask):
        # Convert to Lab color space
        Oai = cv2.cvtColor(Oai, cv2.COLOR_BGR2Lab)
        Obi = cv2.cvtColor(Obi, cv2.COLOR_BGR2Lab)

        # Assemble minimum error image
        minErrorImg = self.assembleMinErrorImg(Oai, Obi, errMask)
        Oai_mean, Oai_std = self.computeMeanAndStd(Oai)
        Obi_mean, Obi_std = self.computeMeanAndStd(Obi)
        minError_mean, minError_std = self.computeMeanAndStd(minErrorImg)
        t_meanVal = 0.5 * 256
        t_stdVal = (1.0 / 36.0) * 256
        t_mean = np.array([t_meanVal] * 3)
        t_std = np.array([t_stdVal] * 3)

        Oai = self.histogramTransform(Oai, Oai_mean, Oai_std, t_mean, t_std)
        Obi = self.histogramTransform(Obi, Obi_mean, Obi_std, t_mean, t_std)

        Oabi = (((0.5 * (Oai + Obi)) - t_meanVal) / 0.5) + t_meanVal
        Oabi_t_mean, Oabi_t_std = self.computeMeanAndStd(Oabi)
        Oabi = self.histogramTransform(Oabi, Oabi_t_mean, Oabi_t_std, minError_mean, minError_std)

        # Convert back to BGR
        Oabi = cv2.cvtColor(Oabi.astype(np.uint8), cv2.COLOR_Lab2BGR)
        Oabi = np.clip(Oabi, 0, 255).astype(np.uint8)
        return Oabi

    def computeMeanAndStd(self, img):
        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))
        return mean, std

    def assembleMinErrorImg(self, Oai, Obi, errMask):
        minErrorImg = Obi.copy()
        minErrorImg[errMask == 1] = Oai[errMask == 1]
        return minErrorImg

    def histogramTransform(self, img, inputMean, inputStd, targetMean, targetStd):
        transformed_img = (img - inputMean) * (targetStd / inputStd) + targetMean
        np.clip(transformed_img, 0, 255, out=img)
        return transformed_img

