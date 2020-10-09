from sklearn.tree import DecisionTreeClassifier

from src.models.model import Model
from src.utils.plot_helper import plot_decision_tree


class DecisionTree(Model):    
    def _get_model_name(self):
        return "Decision Tree"

    def _set_model_type(self):
        return DecisionTreeClassifier(max_depth=3)

    def optional_pipeline(self):
        plot_decision_tree(self._model, idx=self._X_columns, classes=self._target_classes)
