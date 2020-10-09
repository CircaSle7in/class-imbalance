from abc import ABC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.utils.plot_helper import make_confusion_matrix, plot_roc_curve


class Model(ABC):
    def __init__(
        self,
        X,
        y,
        X_columns,
        target_classes,
        # analysis_title,
        data_set_name,
        resample_type=None,
    ):
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, train_size=0.6)
        self.model_type = self._set_model_type()
        self._model = self.model_type.fit(self._X_train, self._y_train)
        self._X_columns = X_columns
        self._target_classes = target_classes
        self._model_name = None
        self._data_set_name = data_set_name
        self._resample_type = resample_type
        # self._analysis_title = analysis_title

    @property
    def __str__(self):
        if self._resample_type:
            return f"{self._get_model_name()} {self._data_set_name} {self._resample_type}"
        else:
            return f"{self._get_model_name()} {self._data_set_name}"

    def _get_model_name(self):
        return

    def _set_model_type(self):
        return

    def main_pipeline(self):
        cf_matrix = confusion_matrix(self._y_test, self._model.predict(self._X_test))
        make_confusion_matrix(cf_matrix, title=self.__str__)
        plot_roc_curve(self._y_test, self._model.predict(self._X_test), title=self.__str__)

    def optional_pipeline(self):
        return

    def run(self):
        print(self.__str__)
        self.main_pipeline()
        self.optional_pipeline()
