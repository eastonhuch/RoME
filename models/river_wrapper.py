import numpy as np
import sklearn
import river
from river.stream import iter_array

class RiverBatchEstimator(sklearn.base.BaseEstimator):
    def __init__(self, model: river.base.Estimator) -> None:
        super().__init__()
        self.model = model

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'RiverBatchEstimator':
        for xi, yi in iter_array(X, y):
            self.model = self.model.learn_one(xi, yi)
        return self

    def predict(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        predictions = [self.model.predict_one(xi, i) for xi, i in iter_array(X, idx)]
        return np.asarray(predictions)
    