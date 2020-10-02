from sklearn.ensemble import RandomForestClassifier

from src.models.model import Model
from src.plot_helper import plot_feature_importance


class RandomForest(Model):
    def _set_model_type(self):
        return RandomForestClassifier()

    def optional_pipeline(self):
        plot_feature_importance(self._model.feature_importances_, self._X_columns)
