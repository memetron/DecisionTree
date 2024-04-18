import numpy as np

# simple class to keep information about how data gets normalized / denormalized
class Normalizer:
    def __init__(self, numFeatures):
        self._normCoefVector = np.ones(numFeatures)
        self._normOffsetVector = np.zeros(numFeatures)
        self._normCoefLabel = 1
        self._normOffsetLabel = 0


    def init_normalization_parameters(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        self._normCoefLabel = 1 / (np.max(expectedVals) - np.min(expectedVals))
        self._normOffsetLabel = (-1) * np.min(expectedVals) * self._normCoefLabel

        self._normCoefVector = 1 / (np.max(featuresArray, axis=0) - np.min(featuresArray, axis=0))
        self._normOffsetVector = (-1) *  np.min(featuresArray, axis=0) * self._normCoefVector

    def normalize_features(self, featuresArray: np.ndarray):
        try:
            return featuresArray * self._normCoefVector + self._normOffsetVector
        except Exception as e:
            print(f"featuresArray = {featuresArray.shape}\n\n"
                  f"self._normCoefVector = {self._normCoefVector.shape}\n\n"
                  f"self._normOffsetVector = {self._normOffsetVector.shape}\n\n")
            raise e

    def normalize_labels(self, labels):
        return labels * self._normCoefLabel + self._normOffsetLabel

    def denormalize_label(self, label):
        return (label - self._normOffsetLabel) / self._normCoefLabel