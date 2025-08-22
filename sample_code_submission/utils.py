import os
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras.utils import plot_model


log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(
        logging, log_level, logging.INFO
    ),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)


def histogram_dataset(
    dfall,
    target,
    weights,
    columns=None,
    nbin=25,
    save_path="histogram.png",
    dataName="",
):
    """
    dfall: numpy array of shape (N, D) — features for N events

    target: numpy array of shape (N,) — labels: 0 = background, 1 = signal

    weights: numpy array of shape (N,) — event weights

    columns: list of feature names (like ["score"])

    nbin: number of histogram bins

    save_path: where to save the final figure

    Plots histograms of the dataset features.
    Draw a histogram for each feature (like score, pt, etc.) and compare how it looks for:


    .. Image:: images/histogram_datasets.png
    """

    if columns is None:
        columns = columns
    else:
        for col in columns:
            if col not in columns:
                logger.warning(f"Column {col} not found in dataset. Skipping.")
                columns.remove(col)
    if len(columns) == 0:
        raise ValueError("No valid columns provided for histogram plotting.")

    sns.set_theme(style="whitegrid")

    df = pd.DataFrame(dfall, columns=columns)

    # Number of rows and columns in the subplot grid
    n_cols = 2  # Number of columns in the subplot grid
    n_rows = int(np.ceil(len(columns) / n_cols))  # Calculate the number of rows needed

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 6 * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

    for i, column in enumerate(columns):
        # Determine the combined range for the current column

        print(f"[*] --- {column} histogram")

        lower_percentile = 0
        upper_percentile = 97.5

        lower_bound = np.percentile(df[column], lower_percentile)
        upper_bound = np.percentile(df[column], upper_percentile)

        df_clipped = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        weights_clipped = weights[
            (df[column] >= lower_bound) & (df[column] <= upper_bound)
        ]
        target_clipped = target[
            (df[column] >= lower_bound) & (df[column] <= upper_bound)
        ]

        min_value = df_clipped[column].min()
        max_value = df_clipped[column].max()

        # Define the bin edges
        bin_edges = np.linspace(min_value, max_value, nbin + 1)

        signal_field = df_clipped[target_clipped == 1][column]
        background_field = df_clipped[target_clipped == 0][column]
        signal_weights = weights_clipped[target_clipped == 1]
        background_weights = weights_clipped[target_clipped == 0]

        # Plot the histogram for label == 1 (Signal)
        axes[i].hist(
            signal_field,
            bins=bin_edges,
            alpha=0.4,
            color="blue",
            label="Signal",
            weights=signal_weights,
            density=True,
        )

        axes[i].hist(
            background_field,
            bins=bin_edges,
            alpha=0.4,
            color="red",
            label="Background",
            weights=background_weights,
            density=True,
        )

        # Set titles and labels
        axes[i].set_title(f"score histogam for{dataName}", fontsize=16)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Density")

        # Add a legend to each subplot
        axes[i].legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    fig.savefig(save_path)
    plt.show()
    plt.close(fig)
    return save_path


def roc_curve_wrapper(
    score,
    labels,
    weights,
    plot_label="model",
    color="b",
    lw=2,
    save_path="roc_curve.png",
):
    """
    Plots the ROC curve.

    Args:
        * score (ndarray): The score.
        * labels (ndarray): The labels.
        * weights (ndarray): The weights.
        * plot_label (str, optional): The plot label. Defaults to "model".
        * color (str, optional): The color. Defaults to "b".
        * lw (int, optional): The line width. Defaults to 2.

    .. Image:: images/roc_curve.png
    """

    auc = roc_auc_score(y_true=labels, y_score=score, sample_weight=weights)

    fig = plt.figure(figsize=(8, 7))

    fpr, tpr, _ = roc_curve(y_true=labels, y_score=score, sample_weight=weights)
    plt.plot(fpr, tpr, color=color, lw=lw, label=plot_label + " AUC :" + f"{auc:.3f}")

    plt.plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.show()
    plt.close(fig)
    return save_path


def visualize_model_architecture(model, filename="nn_architecture.png"):
    """
    Saves a visual diagram of a Keras model architecture to a file.

    Args:
        model (tf.keras.Model): The compiled Keras model.
        filename (str): Path to save the image (e.g., 'model.png').
    """
    try:
        plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
        print(f"[✔] Saved model visualization to: {filename}")
    except ImportError as e:
        print(
            "[✘] Error: Missing 'pydot' or 'graphviz'. Please install them to visualize the model."
        )
        print(str(e))


def stacked_histogram(
    dfall,  # DataFrame containing the full dataset
    target,  # NumPy array: binary labels (0=background, 1=signal)
    weights,  # NumPy array: weights for each event
    detailed_label,  # NumPy array: multi-class labels (e.g. 'htautau', 'ttbar', etc.)
    field_name,  # String: name of the feature column to plot
    mu_hat=1.0,  # Float: scale factor for the signal histogram
    nbins=30,  # Int: number of bins in the histogram
    y_scale="linear",  # String: y-axis scale ('linear' or 'log')
    save_path="stacked_histogram.png",  # String: path where the figure will be saved
):
    """
    Plots and saves a stacked histogram for a given feature (field_name) from the dataset.
    (A stacked histogram is a way to compare multiple groups in one histogram, by stacking their values on top of each other.)

    Args:
        dfall (pd.DataFrame): The full dataset containing the feature columns.
        target (np.ndarray): Binary class labels (0=background, 1=signal).
        weights (np.ndarray): Event weights corresponding to each row in dfall.
        detailed_label (np.ndarray): More specific labels for background types (e.g., 'ttbar', 'ztautau', etc.).
        field_name (str): The name of the feature column to be plotted.
        mu_hat (float, optional): Scale factor for signal strength. Default is 1.0.
        nbins (int, optional): Number of histogram bins. Default is 30.
        y_scale (str, optional): Type of scale to use on the y-axis ('linear' or 'log'). Default is 'linear'.
        save_path (str, optional): Path to save the generated plot image. Default is 'stacked_histogram.png'.

    Returns:
        str: Path where the image was saved.
    """
    field = dfall[field_name]

    weight_keys = {}
    keys = np.unique(detailed_label)

    for key in keys:
        weight_keys[key] = weights[detailed_label == key]

    print("keys", keys)
    print("keys 2", weight_keys.keys())

    sns.set_theme(rc={"figure.figsize": (8, 7)}, style="whitegrid")

    lower_percentile = 0
    upper_percentile = 97.5

    lower_bound = np.percentile(field, lower_percentile)
    upper_bound = np.percentile(field, upper_percentile)

    field_clipped = field[(field >= lower_bound) & (field <= upper_bound)]
    weights_clipped = weights[(field >= lower_bound) & (field <= upper_bound)]
    target_clipped = target[(field >= lower_bound) & (field <= upper_bound)]
    detailed_labels_clipped = detailed_label[
        (field >= lower_bound) & (field <= upper_bound)
    ]

    min_value = field_clipped.min()
    max_value = field_clipped.max()

    # Define the bin edges
    bins = np.linspace(min_value, max_value, nbins + 1)

    """
    hist_s and hist_b are arrays length nbins. They are weighted counts per bin for signal and background respectively.
    """
    hist_s, bins = np.histogram(
        field_clipped[target_clipped == 1],
        bins=bins,
        weights=weights_clipped[target_clipped == 1],
    )

    hist_b, bins = np.histogram(
        field_clipped[target_clipped == 0],
        bins=bins,
        weights=weights_clipped[target_clipped == 0],
    )

    #     hist_bkg preserves the full background histogram for later plotting of the signal+background overlays.
    hist_bkg = hist_b.copy()

    higgs = "htautau"

    for key in keys:
        if key != higgs:
            hist, bins = np.histogram(
                field_clipped[detailed_labels_clipped == key],
                bins=bins,
                weights=weights_clipped[detailed_labels_clipped == key],
            )
            plt.stairs(hist_b, bins, fill=True, label=f"{key} bkg")
            hist_b -= hist
        else:
            print(key, hist_s.shape)

    plt.stairs(
        hist_s * mu_hat + hist_bkg,
        bins,
        fill=False,
        color="orange",
        label=f"$H \\rightarrow \\tau \\tau (\\mu = {mu_hat:.3f})$",
    )

    plt.stairs(
        hist_s + hist_bkg,
        bins,
        fill=False,
        color="red",
        label=f"$H \\rightarrow \\tau \\tau (\\mu = {1.0:.3f})$",
    )

    plt.legend()
    plt.title(f"Stacked histogram of {field_name}")
    plt.xlabel(f"{field_name}")
    plt.ylabel("Weighted count")
    plt.yscale(y_scale)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()
    return save_path


import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep  # Assuming you're using ATLAS/MPLHEP style
import os


def abline(slope, intercept, **kwargs):
    """Plot a line y = slope * x + intercept across the current plot."""
    ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, **kwargs)


def fill_histograms_wError(data, weights, edges, histrange, epsilon=1.0e-20):
    """
    Compute weighted histogram and its statistical error.

    Args:
        data (np.ndarray): Data points to histogram.
        weights (np.ndarray): Weights for each data point.
        edges (np.ndarray): Bin edges.
        histrange (tuple): Range of histogram (min, max).
        epsilon (float): Small number to avoid division by zero.

    Returns:
        hist (np.ndarray): Weighted histogram counts.
        hist_err (np.ndarray): Statistical errors per bin.
    """

    # Weighted histogram
    hist = np.histogram(data, bins=edges, range=histrange, weights=weights)[0]

    # For the error, assuming weights are event weights, the variance per bin is sum of squared weights
    # Compute sum of squared weights in each bin:
    squared_weights = weights**2
    hist_err = np.sqrt(
        np.histogram(data, bins=edges, range=histrange, weights=squared_weights)[0]
    )

    # Avoid zero errors by adding epsilon
    hist_err = np.maximum(hist_err, epsilon)

    return hist, hist_err


"""
Calibration curves check how well your model’s predicted probabilities correspond to true probabilities of the signal (Higgs) events.
"""
# def plot_calibration_curve(

#     # --- Training set inputs ---
#     data_den,      # Array of predicted scores for background events (label=0) in training set
#     weight_den,    # Corresponding event weights for background events in training set
#     data_num,      # Array of predicted scores for signal events (label=1) in training set
#     weight_num,    # Corresponding event weights for signal events in training set

#     # --- Holdout set inputs ---
#     data_denH,     # Array of predicted scores for background events (label=0) in holdout set
#     weight_denH,   # Corresponding event weights for background events in holdout set
#     data_numH,     # Array of predicted scores for signal events (label=1) in holdout set
#     weight_numH,   # Corresponding event weights for signal events in holdout set

#     # --- Plot configuration ---
#     path_to_figures="",    # Directory path to save the plot (not used directly here since `save` overrides)
#     nbins=100,             # Number of bins for score histograms
#     epsilon=1.0e-20,       # Small value to avoid division by zero
#     label="Calibration Curve", # Title label for the plot
#     score_range="standard",    # Range type for score normalization (currently unused in code)
#     save="",                  # Full file path to save the output figure
# ):
#     # Prepare save path
#     save_path = save

#     data = np.concatenate([data_num, data_den, data_denH, data_numH]).flatten()
#     xmin, xmax = np.amin(data), np.amax(data)
#     edges = np.linspace(xmin, xmax, nbins + 1)
#     histrange = (xmin, xmax)

#     # Compute histograms
#     '''
#     h_S and h_X are weighted histograms of background and signal scores on the training set.

#     h_ratio = fraction of signal in each score bin (signal / (signal+background)).

#     Errors are computed using weighted statistical uncertainties from the squared weights (see fill_histograms_wError).

#     Same is done for holdout set: h_SH, h_XH, h_ratioH.
#     '''
#     h_S, h_S_err = fill_histograms_wError(
#         data_den, weight_den, edges, histrange, epsilon
#     )
#     h_X, h_X_err = fill_histograms_wError(
#         data_num, weight_num, edges, histrange, epsilon
#     )
#     h_sum = h_S + h_X
#     h_ratio = np.divide(h_X, h_sum + epsilon)  # avoids divide by zero
#     err = np.sqrt((h_X_err / (h_X + epsilon)) ** 2 + (h_S_err / (h_S + epsilon)) ** 2)

#     h_SH, h_SH_err = fill_histograms_wError(
#         data_denH, weight_denH, edges, histrange, epsilon
#     )
#     h_XH, h_XH_err = fill_histograms_wError(
#         data_numH, weight_numH, edges, histrange, epsilon
#     )
#     h_sumH = h_SH + h_XH
#     h_ratioH = np.divide(h_XH, h_sumH + epsilon)
#     errH = np.sqrt(
#         (h_XH_err / (h_XH + epsilon)) ** 2 + (h_SH_err / (h_SH + epsilon)) ** 2
#     )

#     # Plotting
#     fig = plt.figure(figsize=(12, 8), constrained_layout=True)
#     ax1 = fig.add_subplot(2, 1, 1)

#     bin_centers = (edges[:-1] + edges[1:]) / 2
#     chi_2 = np.sum(((h_ratio - bin_centers) ** 2) / (err**2 + epsilon)) / (nbins - 1)

#     chi_H = np.sum(((h_ratioH - bin_centers) ** 2) / (errH**2 + epsilon)) / (nbins - 1)
#     hep.histplot(
#         [h_ratio, h_ratioH], bins=edges, yerr=[err, errH], label=["Training", "Holdout"]
#     )
#     abline(1.0, 0.0)
#     plt.title(label, fontsize=12)
#     plt.text(x=0.2, y=1.0, s=r"${\chi}^2_{\rm Train}/n_{\rm dof}$ = " + f"{chi_2:.3f}")
#     plt.text(
#         x=0.2, y=0.9, s=r"${\chi}^2_{\rm Holdout}/n_{\rm dof}$ = " + f"{chi_H:.3f}"
#     )
#     plt.axis(xmin=xmin - 0.1, xmax=xmax + 0.1, ymax=1.1, ymin=-0.1)
#     plt.ylabel("Probability ratio", size=12)
#     plt.legend(loc="lower right")

#     # Residuals
#     ax2 = fig.add_subplot(2, 1, 2)
#     slopeOne = (edges[:-1] + edges[1:]) / 2
#     residue = np.divide(h_ratio - slopeOne, err + epsilon)
#     residueH = np.divide(h_ratioH - slopeOne, errH + epsilon)

#     plt.errorbar(slopeOne, residue, yerr=1.0, fmt="o", label="Train Residuals")
#     plt.errorbar(slopeOne, residueH, yerr=1.0, fmt="o", label="Holdout Residuals")
#     plt.xlabel("Predicted Score", size=12)
#     plt.ylabel("Residual", size=12)
#     plt.axis(xmin=xmin - 0.1, xmax=xmax + 0.1, ymin=-4.0, ymax=4.0)
#     abline(0.0, 0.0)
#     plt.legend()

#     # Save figure
#     plt.savefig(save_path, dpi=100, bbox_inches="tight")
#     plt.show()
#     plt.close(fig)
#     return save_path


def plot_calibration_curve(
    data_den,
    weight_den,
    data_num,
    weight_num,
    data_denH,
    weight_denH,
    data_numH,
    weight_numH,
    path_to_figures="",
    nbins=100,
    epsilon=1.0e-20,
    label="Calibration Curve",
    score_range="standard",
    save="",
):

    save_path = save

    # Combine scores for axis limits
    data = np.concatenate([data_num, data_den, data_denH, data_numH]).flatten()
    xmin, xmax = np.amin(data), np.amax(data)
    edges = np.linspace(xmin, xmax, nbins + 1)
    histrange = (xmin, xmax)

    # --- Histograms ---
    h_S, h_S_err = fill_histograms_wError(
        data_den, weight_den, edges, histrange, epsilon
    )
    h_X, h_X_err = fill_histograms_wError(
        data_num, weight_num, edges, histrange, epsilon
    )
    h_sum = h_S + h_X
    h_ratio = np.divide(h_X, h_sum + epsilon)
    err = np.sqrt((h_X_err / (h_X + epsilon)) ** 2 + (h_S_err / (h_S + epsilon)) ** 2)

    h_SH, h_SH_err = fill_histograms_wError(
        data_denH, weight_denH, edges, histrange, epsilon
    )
    h_XH, h_XH_err = fill_histograms_wError(
        data_numH, weight_numH, edges, histrange, epsilon
    )
    h_sumH = h_SH + h_XH
    h_ratioH = np.divide(h_XH, h_sumH + epsilon)
    errH = np.sqrt(
        (h_XH_err / (h_XH + epsilon)) ** 2 + (h_SH_err / (h_SH + epsilon)) ** 2
    )

    # --- Chi-squared ---
    bin_centers = (edges[:-1] + edges[1:]) / 2
    chi_2 = np.sum(((h_ratio - bin_centers) ** 2) / (err**2 + epsilon)) / (nbins - 1)
    chi_H = np.sum(((h_ratioH - bin_centers) ** 2) / (errH**2 + epsilon)) / (nbins - 1)

    # --- Brier score ---
    # Training set
    preds_train = np.concatenate([data_num, data_den])
    labels_train = np.concatenate([np.ones_like(data_num), np.zeros_like(data_den)])
    weights_train = np.concatenate([weight_num, weight_den])
    brier_train = np.average((preds_train - labels_train) ** 2, weights=weights_train)

    # Holdout set
    preds_holdout = np.concatenate([data_numH, data_denH])
    labels_holdout = np.concatenate([np.ones_like(data_numH), np.zeros_like(data_denH)])
    weights_holdout = np.concatenate([weight_numH, weight_denH])
    brier_holdout = np.average(
        (preds_holdout - labels_holdout) ** 2, weights=weights_holdout
    )

    # --- Plot ---
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    ax1 = fig.add_subplot(2, 1, 1)
    hep.histplot(
        [h_ratio, h_ratioH], bins=edges, yerr=[err, errH], label=["Training", "Holdout"]
    )
    abline(1.0, 0.0)
    plt.title(label, fontsize=12)

    # Text metrics
    plt.text(x=0.2, y=1.0, s=rf"${{\chi}}^2_{{\rm Train}}/n_{{\rm dof}}$ = {chi_2:.3f}")
    plt.text(
        x=0.2, y=0.92, s=rf"${{\chi}}^2_{{\rm Holdout}}/n_{{\rm dof}}$ = {chi_H:.3f}"
    )
    plt.text(x=0.2, y=0.84, s=rf"Brier (Train) = {brier_train:.4f}")
    plt.text(x=0.2, y=0.76, s=rf"Brier (Holdout) = {brier_holdout:.4f}")

    plt.axis(xmin=xmin - 0.1, xmax=xmax + 0.1, ymax=1.1, ymin=-0.1)
    plt.ylabel("Probability ratio", size=12)
    plt.legend(loc="lower right")

    # Residuals
    ax2 = fig.add_subplot(2, 1, 2)
    slopeOne = (edges[:-1] + edges[1:]) / 2
    residue = np.divide(h_ratio - slopeOne, err + epsilon)
    residueH = np.divide(h_ratioH - slopeOne, errH + epsilon)

    plt.errorbar(slopeOne, residue, yerr=1.0, fmt="o", label="Train Residuals")
    plt.errorbar(slopeOne, residueH, yerr=1.0, fmt="o", label="Holdout Residuals")
    plt.xlabel("Predicted Score", size=12)
    plt.ylabel("Residual", size=12)
    plt.axis(xmin=xmin - 0.1, xmax=xmax + 0.1, ymin=-4.0, ymax=4.0)
    abline(0.0, 0.0)
    plt.legend()

    # Save
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return save_path


def plot_score_distributions(
    training_set, holdout_set, preselection, columns, models, plots_dir, NP
):
    """
    Plots the score distributions for the nominal, plus, and minus models.

    Args:
        training_set (dict): The training set (should include 'data', 'labels', 'weights')
        holdout_set (dict): The holdout set (should include 'data', 'labels', 'weights')

    Saves:
        A matplotlib figure comparing the score distributions.
    """

    # Apply preselection and extract the required data
    training_set = preselection.apply_pre_selection(training_set, threshold=0.8)
    holdout_set = preselection.apply_pre_selection(holdout_set, threshold=0.8)

    X_train = training_set["data"][columns]
    X_holdout = holdout_set["data"][columns]

    # Predict scores for each systematic model
    scores = {}
    for key in models:
        scores[key] = {
            "train": models[key].predict(X_train),
            "holdout": models[key].predict(X_holdout),
        }

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    colors = {"plus": "red", "minus": "blue", "nominal": "black"}

    for key, color in colors.items():
        if key == "nominal":
            # Estimate nominal as midpoint if not explicitly trained
            scores["nominal"] = {
                "train": 0.5 * (scores["plus"]["train"] + scores["minus"]["train"]),
                "holdout": 0.5
                * (scores["plus"]["holdout"] + scores["minus"]["holdout"]),
            }

        axs[0].hist(
            scores[key]["train"],
            bins=50,
            alpha=0.6,
            label=key,
            color=color,
            density=True,
        )

        axs[1].hist(
            scores[key]["holdout"],
            bins=50,
            alpha=0.6,
            label=key,
            color=color,
            density=True,
        )

    axs[0].set_title("Score Distribution - Training Set")
    axs[1].set_title("Score Distribution - Holdout Set")

    for ax in axs:
        ax.set_xlabel("Model Score")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()

    save_path = os.path.join(plots_dir, f"{NP}_score_comparison.png")
    fig.savefig(save_path)
    plt.show()
    plt.close(fig)
    logger.info(f"Score distribution plot saved at {save_path}")
    return save_path


def plot_two_score_distributions(
    nominal_scores, sys_scores, bins=50, range=(0, 1), save_path=None, type=None
):
    """
    Plot score distributions for nominal, plus, and minus systematic variations on the same plot.

    Args:
        nominal_scores (np.array): classifier scores for nominal events
        plus_scores (np.array): classifier scores for plus variation events
        minus_scores (np.array): classifier scores for minus variation events
        bins (int): number of bins in the histogram
        range (tuple): min and max score values to plot
        save_path (str): optional path to save the plot

    Returns:
        None
    """

    plt.figure(figsize=(10, 6))

    plt.hist(
        nominal_scores,
        bins=bins,
        range=range,
        histtype="step",
        linewidth=2,
        label="Nominal",
        color="black",
        density=True,
    )

    plt.hist(
        sys_scores,
        bins=bins,
        range=range,
        histtype="step",
        linewidth=2,
        label=f"{type} variation",
        color="red",
        density=True,
    )

    plt.xlabel("Classifier Score")
    plt.ylabel("Normalized Counts")
    plt.title(f"Score Distributions for Nominal and {type} Systematic Variations")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()
    return save_path


def plot_score_distribution_for_triDataset(
    nominal_scores,
    plus_scores,
    minus_scores,
    bins=50,
    range=(0, 1),
    save_path=None,
    type=None,
):
    """
    Plot score distributions for nominal, plus, and minus systematic variations on the same plot.

    Args:
        nominal_scores (np.array): classifier scores for nominal events
        plus_scores (np.array): classifier scores for plus variation events
        minus_scores (np.array): classifier scores for minus variation events
        bins (int): number of bins in the histogram
        range (tuple): min and max score values to plot
        save_path (str): optional path to save the plot

    Returns:
        None
    """

    plt.figure(figsize=(10, 6))

    plt.hist(
        nominal_scores,
        bins=bins,
        range=range,
        histtype="step",
        linewidth=2,
        label="Nominal",
        color="black",
        density=True,
    )

    plt.hist(
        plus_scores,
        bins=bins,
        range=range,
        histtype="step",
        linewidth=2,
        label="Plus variation",
        color="red",
        density=True,
    )

    plt.hist(
        minus_scores,
        bins=bins,
        range=range,
        histtype="step",
        linewidth=2,
        label="Minus variation",
        color="blue",
        density=True,
    )

    plt.xlabel("Classifier Score")
    plt.ylabel("Normalized Counts")
    plt.title(
        f"Score Distributions for Nominal , Plus and Minus Systematic Variations for {type} Model"
    )
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()
    return save_path


def plot_three_systematics_calibration(
    nominal_scores,
    nominal_labels,
    nominal_weights,
    plus_scores,
    plus_labels,
    plus_weights,
    minus_scores,
    minus_labels,
    minus_weights,
    bins=50,
    save_path=None,
):
    """
    Plot calibration curves for nominal, plus, and minus systematic variations on the same plot.

    Args:
        nominal_scores, plus_scores, minus_scores: np.arrays of classifier scores
        nominal_labels, plus_labels, minus_labels: np.arrays of 0/1 labels (nominal=0, shifted=1)
        nominal_weights, plus_weights, minus_weights: np.arrays of event weights
        bins: number of bins for histogram
        save_path: filepath to save the plot (optional)

    Returns:
        None
    """

    def compute_fraction_shifted(scores, labels, weights, bins):
        edges = np.linspace(np.min(scores), np.max(scores), bins + 1)
        fraction_shifted = []
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        for i in range(bins):
            mask = (scores >= edges[i]) & (scores < edges[i + 1])
            if np.sum(weights[mask]) == 0:
                fraction_shifted.append(0)
            else:
                weighted_shifted = np.sum(weights[mask & (labels == 1)])
                weighted_total = np.sum(weights[mask])
                fraction_shifted.append(weighted_shifted / weighted_total)
        return bin_centers, fraction_shifted

    x_nom, y_nom = compute_fraction_shifted(
        nominal_scores, nominal_labels, nominal_weights, bins
    )
    x_plus, y_plus = compute_fraction_shifted(
        plus_scores, plus_labels, plus_weights, bins
    )
    x_minus, y_minus = compute_fraction_shifted(
        minus_scores, minus_labels, minus_weights, bins
    )

    plt.figure(figsize=(10, 6))
    plt.plot(x_nom, y_nom, label="Nominal", color="black")
    plt.plot(x_plus, y_plus, label="Plus Variation", color="red")
    plt.plot(x_minus, y_minus, label="Minus Variation", color="blue")
    plt.xlabel("Classifier Score")
    plt.ylabel("Fraction Shifted Events")
    plt.title("Calibration Curve: Nominal vs Plus vs Minus Systematic Variations")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()
    return save_path
