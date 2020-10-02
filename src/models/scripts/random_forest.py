import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.data_helper import create_dataset
from src.plot_helper import make_confusion_matrix, plot_roc_curve, plot_feature_importance


def pipeline(model, X, y, X_columns, y_classes):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    model.fit(X_train, y_train)

    cf_matrix = confusion_matrix(y_test, model.predict(X_test))
    make_confusion_matrix(cf_matrix)

    plot_roc_curve(y_test, model.predict(X_test))

    plot_feature_importance(model.feature_importances_, X_columns)


def main(X, y, X_columns, analysis_title):
    print(analysis_title)
    model = RandomForestClassifier()
    print(model)
    pipeline(model, X, y, X_columns, ['majority', 'minority'])


if __name__ == "__main__":
    main()
