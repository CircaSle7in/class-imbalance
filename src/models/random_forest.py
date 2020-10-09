from sklearn.ensemble import RandomForestClassifier

from src.models.model import Model
from src.utils.plot_helper import plot_feature_importance


class RandomForest(Model):
    def _get_model_name(self):
        return "Random Forest"

    def _set_model_type(self):
        return RandomForestClassifier()

    def optional_pipeline(self):
        plot_feature_importance(
            self._model.feature_importances_,
            self._X_columns,
            title=self.__str__
        )
