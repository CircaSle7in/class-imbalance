from collections import Counter
import numpy as np
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from IPython.display import display, Image
from matplotlib import pyplot as plt
import pandas as pd
import pydotplus
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC
from sklearn.tree import export_graphviz

from src.utils.data_helper import apply_resampler


def plot_resampling(X_res, y_res, sampling, ax):
    # X_res, y_res = sampling.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y_res)


def plot_decision_function(X, y, clf, ax):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step),
        np.arange(y_min, y_max, plot_step)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')


def plot_everything(
        X,
        y,
        save_fig=False,
        file_path=None
):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 25))
    ax_arr = ((ax1, ax2), (ax3, ax4))

    resampled_data = {
        'SMOTE': {
            'X_res': None,
            'y_res': None,
        },
        'ADASYN': {
            'X_res': None,
            'y_res': None,
        },
    }

    for ax, sampler in zip(ax_arr, (SMOTE(random_state=0), ADASYN(sampling_strategy="minority", random_state=0))):
        clf = make_pipeline(sampler, LinearSVC())
        clf.fit(X, y)

        plot_decision_function(X, y, clf, ax[0])
        ax[0].set_title(f'Decision function for {sampler.__class__.__name__}')

        X_res, y_res = apply_resampler(X, y, sampler)

        plot_resampling(X_res, y_res, sampler, ax[1])
        ax[1].set_title(f'Resampling using {sampler.__class__.__name__}')

        resampled_data[sampler.__class__.__name__]['X_res'] = X_res
        resampled_data[sampler.__class__.__name__]['y_res'] = y_res

    fig.tight_layout()

    if save_fig:
        plt.savefig(file_path)

    plt.show()

    return resampled_data


def plot_feature_importance(feature_importance, idx: list, title):
    feature_importances = pd.DataFrame(
        feature_importance,
        index=idx,
        columns=['importance']).sort_values('importance', ascending=False)
    plt.barh(feature_importances.index, feature_importances.importance)
    plt.title(title)
    plt.show()


def plot_decision_tree(model, idx: list, classes: list):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=idx,
        class_names=classes,
        rounded=True,
        filled=True,
    )
    graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    img = Image(graph.create_png())
    display(img)


def make_confusion_matrix(
        cf,
        group_names=None,
        categories='auto',
        count=True,
        percent=True,
        cbar=True,
        xyticks=True,
        xyplotlabels=True,
        sum_stats=True,
        figsize=None,
        cmap='Blues',
        title=None
):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = [f"{value}\n" for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = [f"{value:0.0f}\n" for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [f"{value:.2%}" for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = (
                f"\n\nAccuracy={accuracy:0.3f}\n"
                f"Precision={precision:0.3f}\n"
                f"Recall={recall:0.3f}\n"
                f"F1 Score={f1_score:0.3f}"
            )
        else:
            stats_text = "\n\nAccuracy={accuracy:0.3f}"
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    plt.show()


def plot_roc_curve(y, y_score, title):
    ns_score = [0 for _ in range(len(y))]
    ns_fpr, ns_tpr, _ = roc_curve(y, ns_score)

    fpr, tpr, _ = roc_curve(y, y_score)

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Fitted')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} ROC Curve (AUC={auc(fpr, tpr):.4f}')
    plt.legend()
    plt.show()

    # fig = px.area(
    #     x=fpr, y=tpr,
    #     title=f'{title} ROC Curve (AUC={auc(fpr, tpr):.4f})',
    #     labels=dict(x='False Positive Rate', y='True Positive Rate'),
    #     width=400, height=400
    # )
    # fig.add_shape(
    #     type='line', line=dict(dash='dash'),
    #     x0=0, x1=1, y0=0, y1=1
    # )
    #
    # fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # fig.update_xaxes(constrain='domain')
    # fig.show()
