import numpy as np
from sys import path
import pickle
import matplotlib.pyplot as plt

import seaborn as snb
import pandas as pd
import logging
import os
import mlflow

log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(
        logging, log_level, logging.INFO
    ),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)

path.append("../")
path.append("../ingestion_program")

current_dir = os.path.dirname(__file__)


def fill_histograms_wError(data, weights, edges, histrange, normalize=True):

    h, _ = np.histogram(data, edges, histrange, weights=weights)

    h_err, _ = np.histogram(data, edges, histrange, weights=weights**2)

    h_eff = (h**2) / (h_err)

    N_eff = (np.sum(weights) ** 2) / (np.sum(weights**2))

    sigma_p = np.sqrt(h_eff * (1 - h_eff) / N_eff)

    if normalize:
        h_sum = np.sum(h)

        h = h / h_sum

        h_err = h_err / (h_sum**2)

    total_error = np.sqrt(h_err + sigma_p**2)

    return h, total_error


def roc_curve(y_tests, y_preds, w_tests, labels=None, filename="roc_curve.png"):
    """
    Plot the ROC curve.

    Args:
        y_test (numpy.ndarray): Array of true labels.
        y_pred (numpy.ndarray): Array of predicted labels.
        w_test (numpy.ndarray): Array of weights.
        filename (str, optional): Name of the output file. Defaults to "roc_curve.png".
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    if labels == None:
        labels = ["" for _ in range(len(y_test))]

    for y_test, y_pred, w_test, label in zip(y_tests, y_preds, w_tests, labels):
        fpr, tpr, _ = roc_curve(y_test, y_pred, sample_weight=w_test)
        auc = roc_auc_score(y_test, y_pred, sample_weight=w_test)
        plt.plot(fpr, tpr, label=f"ROC curve {label}(AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], color="black", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()


def box_plot(predictions, variances, name="", bins=10):

    predictions = np.array(predictions)[:2000]
    variances = np.array(variances)[:2000]

    xmin = min(predictions)
    xmax = max(predictions)

    edges = np.linspace(xmin, xmax, bins + 1)

    bin_data = {}
    for i in range(bins):
        # Define the range for filtering
        lower_bound = edges[i]
        upper_bound = edges[i + 1]  # Ranges from 0.1 to 0.9 in steps of 0.1

        # Define the condition for filtering
        condition = (predictions >= lower_bound) & (predictions < upper_bound)
        # Apply the condition to variances and assign to List_Box
        bin_data[f"{((lower_bound + upper_bound)/2):.1f}"] = variances[condition]

    List_Box_df = pd.DataFrame.from_dict(bin_data, orient="index").T

    sb = snb.boxplot(data=List_Box_df)
    sb.set_xlabel(name, fontsize=14)
    sb.set_ylabel("Variance", fontsize=14)
    plt.savefig(current_dir + f"/plots/{name}_v_var_box.png")
    plt.close()


def check_calibration(
    score, y_test, w_test, bins=50, filename="calibration.png", label="Calibrated"
):
    logger.info("Checking calibration")
    plt.figure()

    xmin = min(score)
    xmax = max(score)

    edges = np.linspace(xmin, xmax, bins + 1)
    histrange = (xmin, xmax)

    score_sig = score[y_test == 1.0]
    weight_sig = w_test[y_test == 1.0]
    score_bkg = score[y_test == 0.0]
    weight_bkg = w_test[y_test == 0.0]

    h_S, h_S_err = fill_histograms_wError(score_sig, weight_sig, edges, histrange)
    h_X, h_X_err = fill_histograms_wError(score_bkg, weight_bkg, edges, histrange)
    h_sum = h_S + h_X
    h_ratio = h_S / h_sum

    err = (
        h_ratio**2
        * np.abs(h_S / h_X)
        * np.sqrt((h_X_err / (h_X) ** 2) + (h_S_err / (h_S) ** 2))
    )

    # err = (1 / h_sum**2) * np.sqrt(h_X**2 * h_S_err**2 + h_X_err**2)

    # Create a figure with two subplots: one for histograms and one for ratio
    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Plot histograms on the first subplot
    bin_centers = 0.5 * (edges[:-1] + edges[1:])  # Calculate bin centers
    ax1.step(bin_centers, h_ratio, where="mid", label=label, color="blue")

    # Add error bars to histograms
    ax1.errorbar(
        bin_centers,
        h_ratio,
        yerr=err,
        fmt="none",
        color="blue",
    )

    ax1.plot([xmin, xmax], [xmin, xmax], color="black", linestyle="--", label="Ideal")

    # Add labels and legend to the first subplot
    ax1.set_ylabel("MC estimated ps/(ps + pb)")
    ax1.legend()
    # ax1.set_yscale("log")

    # Residual plot
    residue = h_ratio - bin_centers

    ax2.hist(
        bin_centers,
        bins=edges,
        weights=residue,
        histtype="step",
        color="black",
        linestyle="--",
        label="Residue",
    )

    ax2.axhline(
        0, color="gray", linestyle="--", linewidth=1
    )  # Add a horizontal line at 1 for reference

    # Add labels and legend to the second subplot
    ax2.set_ylabel("residue")
    ax2.set_xlabel("Predicted Score")
    ax2.legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    plt.savefig(filename)
    plt.close()


def check_reweighting(density_ratio, data_set, name="", num=25):
    """
    Perform calculations to calculate mu.

    Args:
        score (numpy.ndarray): Array of scores.
        weight (numpy.ndarray): Array of weights.
        saved_info (dict): Dictionary containing saved information.

    Returns:
        dict: Dictionary containing calculated values of mu_hat, del_mu_stat, del_mu_sys, and del_mu_tot.
    """
    logger.info("Checking reweighting")

    label = data_set["labels"]
    data = data_set["data"]
    weights = data_set["weights"]

    weights_sig = weights[label == 1]
    weights_bkg = weights[label == 0]

    weights_reweighted = weights_sig * density_ratio[label == 1]
    logger.debug(f"Shape of data: {data.shape}")
    logger.debug(f"Shape of label: {label.shape}")
    logger.debug(f"Shape of weights: {weights.shape}")
    logger.debug(f"Shape of density_ratio: {density_ratio.shape}")

    paths = []

    for col in data.columns:
        sig_field = data[col][label == 1]
        bkg_field = data[col][label == 0]

        concat = np.concatenate([sig_field, bkg_field]).flatten()
        xmin = np.amin(concat)
        xmax = np.amax(concat)

        edges = np.linspace(xmin, xmax, num=num + 1)
        histrange = (xmin, xmax)

        '''
        h_S: histogram of signal with original weights.

        h_X: histogram of background with original weights.

        h_S_rw: histogram of signal with reweighted weights.
        '''
        h_S, h_S_err = fill_histograms_wError(sig_field, weights_sig, edges, histrange)
        h_X, h_X_err = fill_histograms_wError(bkg_field, weights_bkg, edges, histrange)
        '''
        The goal of reweighting is to make the signal distribution look more like the background (or to correct for systematic shifts).
        The density_ratio tells you how much each signal event should be up- or down-weighted.
        By multiplying the original signal weights by this ratio, you shift the weighted distribution of signal events.
        In practice:
        Signal events with scores closer to background-like will have larger density ratios (since (1-score)/score is bigger if score is small), so their weight will increase.
        ​Signal events with scores confidently signal-like (score near 1) get smaller density ratios, so their weight decreases.
        '''
        h_S_rw, h_S_rw_err = fill_histograms_wError(
            sig_field, weights_reweighted, edges, histrange
        )

        # Calculate ratio (S - B_rw)
        ratio = h_S / h_S_rw

        # Create a figure with two subplots: one for histograms and one for ratio
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )

        # Plot histograms on the first subplot
        bin_centers = 0.5 * (edges[:-1] + edges[1:])  # Calculate bin centers
        ax1.hist(
            bin_centers,
            bins=edges,
            weights=h_S,
            histtype="step",
            color="blue",
            label="Signal",
        )
        ax1.hist(
            bin_centers,
            bins=edges,
            weights=h_X,
            histtype="step",
            color="green",
            label="Background",
        )
        ax1.hist(
            bin_centers,
            bins=edges,
            weights=h_S_rw,
            histtype="step",
            color="red",
            label="Reweighted S -> B",
        )

        # Add error bars to histograms
        ax1.errorbar(bin_centers, h_S, yerr=h_S_err, fmt="none", color="blue")
        ax1.errorbar(bin_centers, h_X, yerr=h_X_err, fmt="none", color="green")
        ax1.errorbar(bin_centers, h_S_rw, yerr=h_S_rw_err, fmt="none", color="red")

        # Add labels and legend to the first subplot
        ax1.set_ylabel("Counts")
        ax1.legend()
        ax1.set_title(col)

        # Plot residuals as a line plot with error bands on the second subplot
        ax2.hist(
            bin_centers,
            bins=edges,
            weights=ratio,
            histtype="step",
            color="black",
            linestyle="--",
            label="Ratio (S - B_rw)",
        )
        ax2.axhline(
            1, color="gray", linestyle="--", linewidth=1
        )  # Add a horizontal line at 1 for reference

        # Add labels and legend to the second subplot
        ax2.set_xlabel("Variable")
        ax2.set_ylabel("ratio")
        ax2.legend()

        # Adjust layout and display the plot
        plt.tight_layout()
        figure_name = f"{current_dir}/plots/{name}_reweighting_{col}.png"
        fig.savefig(figure_name)
        plt.show()
        plt.close(fig)
        paths.append(figure_name)
    return paths


def plot_NLL(mu, NLL, filename="NLL.png"):
    """
    Plot the negative log-likelihood function.

    Args:
        mu (numpy.ndarray): Array of mu values.
        NLL (numpy.ndarray): Array of negative log-likelihood values.
        filename (str, optional): Name of the output file. Defaults to "NLL.png".
    """

    plt.figure()
    plt.plot(mu, NLL, color="blue")
    plt.vlines(
        mu[np.argmin(NLL)],
        np.min(NLL),
        np.max(NLL),
        color="red",
        linestyle="--",
        label=r"$\hat{\mu}$",
    )
    plt.vlines(
        1, np.min(NLL), np.max(NLL), color="black", linestyle="--", label=r"$\mu = 1$"
    )
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$t_{\mu}$")
    plt.title("Negative Log-Likelihood Function")
    plt.savefig(filename)
    plt.close()


def stacked_histogram(
    detailed_label,
    field,
    labels,
    weights,
    pseudo_weight=None,
    pseudo_field=None,
    mu_hat=1.0,
    bins=30,
    y_scale="linear",
    plot_label="stacked_histogram",
):
    """
    Plots a stacked histogram of a specific field in the dataset.

    Args:
        * field (numpy.ndarray): The field to plot.
        * labels (numpy.ndarray): The labels of the dataset.
        * weights (numpy.ndarray): The weights of the dataset.
        * mu_hat (float): The value of mu_hat.
        * bins (int): The number of bins.
        * y_scale (str): The scale of the y-axis.
        * plot_label (str): The label of the plot.


    .. Image:: ../images/stacked_histogram.png
    """

    bins = 30

    hist_s, bins = np.histogram(
        field[labels == 1],
        bins=bins,
        weights=weights[labels == 1],
    )

    hist_b, bins = np.histogram(
        field[labels == 0],
        bins=bins,
        weights=weights[labels == 0],
    )

    hist_bkg = hist_b.copy()

    categories = np.unique(detailed_label)

    categories = [cat for cat in categories if cat != "htautau"]

    plt.figure()

    # Iterate over each category (excluding "htautau")
    for category in categories:
        condition = detailed_label == category
        field_key = np.array(field[condition])
        labels_key = np.array(labels[condition])
        weights_key = np.array(weights[condition])
        hist, bins = np.histogram(
            field_key,
            bins=bins,
            weights=weights_key,
        )
        plt.stairs(hist_b, bins, fill=True, label=f"{category} bkg")
        hist_b -= hist

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

    if (pseudo_weight is not None) and (pseudo_field is not None):
        hist_pseudo, bins = np.histogram(
            pseudo_field,
            bins=bins,
            weights=pseudo_weight,
        )

        err_pseudo = np.histogram(
            pseudo_field,
            bins=bins,
            weights=pseudo_weight**2,
        )[0]

        plt.errorbar(
            0.5 * (bins[1:] + bins[:-1]),
            hist_pseudo,
            yerr=np.sqrt(err_pseudo),
            fmt="o",
            color="black",
            label="Pseudo data",
        )

    plt.legend()
    plt.title(f"Stacked histogram: {plot_label}")
    plt.xlabel(f"{plot_label}")
    plt.ylabel("Weighted count")
    plt.yscale(y_scale)
    plt.savefig(f"{current_dir}/plots/{plot_label}.png")
    plt.show()
    plt.close()


def simple_plot(
    x,
    y,
    x_label="",
    y_label="",
    title="",
    save=False,
    close=True,
    filename="simple_plot",
):

    plt.plot(x, y, "o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if save:
        plt.savefig(f"{current_dir}/plots/{filename}.png")
    if close:
        plt.show()
        plt.close()
