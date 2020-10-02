import numpy as np
from sklearn.ensemble import BaggingClassifier

from src.models.model import Model
from src.plot_helper import plot_feature_importance


class BaggingModel(Model):
    def _set_model_type(self):
        return BaggingClassifier()

    def optional_pipeline(self):
        plot_feature_importance(self._compute_feature_importance(), self._X_columns)

    def _compute_feature_importance(self):
        feature_importance = np.mean([tree.feature_importances_ for tree in self._model.estimators_], axis=0)
        return feature_importance
