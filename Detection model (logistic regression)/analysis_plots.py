import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    plot_roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
)
from imblearn.over_sampling import RandomOverSampler

color_palette = sns.color_palette("hls", 8)
colormap = ListedColormap(sns.color_palette("hls", 8).as_hex())


def plot_feature_importance(ax, classifier, X_test, y_test, feature_names):
    """Box plot of permutation feature importance"""
    importance = permutation_importance(classifier, X_test, y_test, n_repeats=20)
    decreasing_importance = (
        pd.DataFrame({"mean": importance["importances_mean"], "label": feature_names})
        .sort_values("mean", ascending=False)
        .label.values
    )

    df = (
        pd.DataFrame(
            columns=pd.Index(data=feature_names, name="features"),
            data=importance["importances"].T,
        )
        .stack("features")
        .reset_index()
    )
    df.columns = ["iter", "features", "importance"]
    sns.boxplot(
        y="features",
        x="importance",
        data=df,
        order=decreasing_importance,
        ax=ax,
        palette=color_palette,
    )


def plot_confusion_matrix_with_threshold(ax, classifier, X_test, y_test, threshold):
    proba = classifier.predict_proba(X_test)
    y_pred = (proba > threshold).T[1]

    cm = confusion_matrix(
        y_test, y_pred, sample_weight=None, labels=None, normalize=None
    )

    display_labels = classifier.classes_

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    plot = disp.plot(
        include_values=True,
        ax=ax,
        xticks_rotation="horizontal",
        values_format=None,
        colorbar=False,
    )

    print(plot.text_[0][0])


def plot_roc_vs_threshold(ax, classifier, X_test, y_test):
    plot_roc_curve(classifier, X_test, y_test, ax=ax)
    ax.set_ylabel("True positive rate")
    ax.set_xlabel("False positive rate")


def plot_analysis(classifier, X_test, y_test, feature_names, threshold):
    x_resampled, y_resampled = RandomOverSampler().fit_resample(X_test, y_test)

    with plt.style.context("seaborn"):
        plt.rcParams.update(
            {
                "figure.facecolor": (1.0, 1.0, 1.0, 0.3),  # white   with alpha = 30%
                "axes.facecolor": (1.0, 1.0, 1.0, 0.5),  # white with alpha = 50%
                "savefig.facecolor": (1.0, 1.0, 1.0, 0.2),  # white  with alpha = 20%
            }
        )
        fig, axes = plt.subplot_mosaic(
            [["left", "upper right", "right"], ["left", "lower right"]],
            figsize=(8, 4),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [2, 1], "height_ratios": [1, 1]},
        )
        PrecisionRecallDisplay().from_estimator(
            classifier,
            x_resampled,
            y_resampled,
            ax=axes["right"],
            name="precision-recall curve",
        )
        plot_roc_vs_threshold(axes["upper right"], classifier, x_resampled, y_resampled)
        plot_feature_importance(
            axes["left"], classifier, x_resampled, y_resampled, feature_names
        )
        plot_confusion_matrix_with_threshold(
            axes["lower right"], classifier, x_resampled, y_resampled, threshold
        )

    return fig
