from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, ADASYN
import uuid



def create_dataset(
        n_samples=1000,
        weights=(0.01, 0.99),
        n_classes=2,
        class_sep=0.8,
        n_clusters=1
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters,
        weights=list(weights),
        class_sep=class_sep,
        random_state=0
    )
    return X, y, [uuid.uuid1(x).hex for x in range(n_classes)]


def apply_resampler(X, y, resampler):
    X_res, y_res = resampler.fit_resample(X, y)
    return X_res, y_res
