import os
import pandas as pd

from src.utils.data_helper import create_dataset
from src.models.bagging import BaggingModel
from src.models.decision_tree import DecisionTree
from src.models.random_forest import RandomForest
from src.utils.plot_helper import plot_everything


def run_pipeline(X, y, X_columns, data_set_name):
    DecisionTree(
        X,
        y,
        X_columns,
        ['majority', 'minority'],
        data_set_name=data_set_name
    ).run()
    BaggingModel(
        X,
        y,
        X_columns,
        ['majority', 'minority'],
        data_set_name=data_set_name
    ).run()
    RandomForest(
        X,
        y,
        X_columns,
        ['majority', 'minority'],
        data_set_name=data_set_name
    ).run()

    resampled_data = plot_everything(X, y)

    for resample in resampled_data:
        BaggingModel(
            resampled_data.get(resample).get("X_res"),
            resampled_data.get(resample).get("y_res"),
            X_columns,
            ['majority', 'minority'],
            data_set_name=data_set_name,
            resample_type=resample
        ).run()
        RandomForest(
            resampled_data.get(resample).get("X_res"),
            resampled_data.get(resample).get("y_res"),
            X_columns,
            ['majority', 'minority'],
            data_set_name=data_set_name,
            resample_type=resample
        ).run()


def main():
    # Synthetic examples
    weights = {
        '75% Skew': (0.75, 0.25),
        '90% Skew': (0.90, 0.10),
        '99% Skew': (0.99, 0.01),
    }

    for key, value in weights.items():
        X, y, X_columns = create_dataset(
            n_samples=10000,
            weights=value,
            n_classes=2,
        )

        run_pipeline(X, y, X_columns, key)

    # Real dataset from thyroid analysis
    path = r'/data/thyroid.csv'

    try:
        thyroid_data = pd.read_csv(path)
    except FileNotFoundError:
        os.chdir('..')
        thyroid_data = pd.read_csv(os.getcwd() + path)

    target = 'Class'
    X = thyroid_data.drop(target, axis=1)
    y = thyroid_data.loc[:, target]

    run_pipeline(X, y, X.columns, 'Thyroid Dataset')


if __name__ == "__main__":
    main()
