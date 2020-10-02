from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.plot_helper import make_confusion_matrix, plot_roc_curve, plot_decision_tree


def pipeline(model, X, y, X_columns, y_classes):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    model.fit(X_train, y_train)

    cf_matrix = confusion_matrix(y_test, model.predict(X_test))

    make_confusion_matrix(cf_matrix)
    plot_roc_curve(y_test, model.predict(X_test))
    plot_decision_tree(model, idx=X_columns, classes=y_classes)


def main(X, y, X_columns, analysis_title):
    print(analysis_title)
    dt = DecisionTreeClassifier(max_depth=3)
    pipeline(dt, X, y, X_columns, ['majority', 'minority'])


if __name__ == "__main__":
    main()
