import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import chi2, norm
from sklearn.metrics import roc_curve, roc_auc_score

# region GOF
def wilson_ci(n_events, n_tot, coverage):
    """
    Compute the Wilson confidence interval for a binomial proportion.

    Parameters
    ----------
    n_events : int
        Number of observed events.

    n_tot : int
        Total number of observations.

    coverage : float
        Confidence level (between 0 and 1).

    Returns
    -------
    tuple of float
        Lower confidence bound, observed proportion, and upper confidence
        bound.
    """
    if n_tot == 0:
        return 0.0, 0.0
    p = n_events / n_tot
    alpha = 1 - coverage
    z = norm.ppf(1 - alpha/2)
    denom = 1 + z**2 / n_tot
    centre = p + z**2 / (2 * n_tot)
    half_width = z * np.sqrt(p*(1 - p) / n_tot + z**2 / (4 * n_tot**2))
    lower = (centre - half_width) / denom
    upper = (centre + half_width) / denom
    return np.max([0, lower]), p, upper

def plot_ci(EUD_subset, labels_subset, color, min_pts, coverage, n_quantiles, ax=None):
    """
    Plot observed event rates and Wilson confidence intervals across
    EUD quantile bins.
    """
    if ax is None:
        ax = plt.gca()
    if len(EUD_subset) < 3:
        return
    bins_edges = np.quantile(EUD_subset, np.linspace(0, 1, n_quantiles + 1))
    bins_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
    for i in range(len(bins_centers)):
        if i == len(bins_centers)-1:
            in_bin = (EUD_subset >= bins_edges[i]) & (EUD_subset <= bins_edges[i+1])
        else:
            in_bin = (EUD_subset >= bins_edges[i]) & (EUD_subset < bins_edges[i+1])
        n_tot = np.sum(in_bin)
        if n_tot >= min_pts:
            N_events = np.sum(labels_subset[in_bin])
            lower, center, upper = wilson_ci(N_events, n_tot, coverage)
            ax.errorbar(
                bins_centers[i], center,
                yerr=[[center-lower],[upper-center]],
                fmt='o', color=color, capsize=5,
                markersize=10, markerfacecolor=color, elinewidth=2
            )
            ax.text(
                bins_centers[i] + 0.6, center,
                f'{int(N_events)}/{int(n_tot)}',
                color=color, fontsize=13, verticalalignment='center'
            )

def int_slope_calibration(labels, predictions, plot_res=False, ax=None):
    """
    Estimate the calibration intercept and slope by logistic recalibration.

    Parameters
    ----------
    labels : array-like
        Observed binary outcomes.

    predictions : array-like
        Predicted event probabilities.

    plot_res : bool, default=False
        If True, display the recalibration fit.

    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the plot.

    Returns
    -------
    tuple of float
        Calibration intercept and slope.
    """
    logit_p = np.log(predictions / (1 - predictions))
    X = sm.add_constant(logit_p)
    model = sm.Logit(labels, X).fit(disp=False)
    intercept, slope = model.params

    if plot_res:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        sorted_idx = np.argsort(logit_p)
        ax.plot(logit_p[sorted_idx], labels[sorted_idx], 'o', alpha=0.5, label='Observed')
        ax.plot(logit_p[sorted_idx], intercept + slope*logit_p[sorted_idx], 'r-', label='Calibrated line')
        ax.set_xlabel('Logit(predictions)')
        ax.set_ylabel('Observed labels')
        ax.set_title('Intercept + Slope Calibration')
        ax.legend()

    return intercept, slope

def calibration_plot(labels, predictions, title='', ax=None, n_bins=5):
    """
    Plot model calibration and compute calibration metrics.

    The plot compares predicted probabilities with observed event
    frequencies across probability bins and reports the calibration
    intercept and calibration slope.

    Parameters
    ----------
    labels : array-like
        Observed binary outcomes.

    predictions : array-like
        Predicted event probabilities.

    title : str, default=""
        Plot title.

    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the plot.

    n_bins : int, default=5
        Number of probability bins.

    Returns
    -------
    tuple
        ``ax`` :
            Matplotlib axes containing the plot.

        ``intercept`` :
            Calibration intercept.

        ``slope`` :
            Calibration slope.
    """
    bins_edges = np.linspace(np.min(predictions), np.max(predictions), n_bins + 1)
    bins_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

    coverage = 0.68
    min_pts = 3
    hist_max_height = 0.5

    obs_points = []
    obs_centers = []
    ci_lowers = []
    ci_uppers = []
    n_events_bin = []
    n_total_bin = []

    all_bins_centers = []
    all_n_events_bin = []
    all_n_total_bin = []

    bin_indices = np.digitize(predictions, bins_edges, right=False) - 1
    bin_indices[bin_indices == n_bins] = n_bins - 1

    for i in range(n_bins):
        in_bin = bin_indices == i
        n_tot = np.sum(in_bin)
        n_events = np.sum(labels[in_bin])

        all_bins_centers.append(bins_centers[i])
        all_n_events_bin.append(n_events)
        all_n_total_bin.append(n_tot)

        if n_tot >= min_pts:
            lower, center, upper = wilson_ci(n_events, n_tot, coverage)
            obs_points.append(center)
            obs_centers.append(bins_centers[i])
            ci_lowers.append(lower)
            ci_uppers.append(upper)
            n_events_bin.append(n_events)
            n_total_bin.append(n_tot)

    scale = hist_max_height / max(all_n_total_bin) if max(all_n_total_bin) > 0 else 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [0, 1], 'k--', label='Ideal calibration')

    if len(obs_centers) > 0:
        ax.errorbar(
            obs_centers,
            obs_points,
            yerr=[np.maximum(np.array(obs_points)-np.array(ci_lowers), 0),
                  np.array(ci_uppers)-np.array(obs_points)],
            fmt='o', color='blue', markersize=8, markerfacecolor='blue',
            elinewidth=2, capsize=4
        )

    for x, ne, nt in zip(all_bins_centers, all_n_events_bin, all_n_total_bin):
        width = (bins_edges[1] - bins_edges[0]) * 0.9
        ax.bar(x, ne * scale, width=width, color='red', align='center')
        ax.bar(x, -(nt - ne) * scale, width=width, color='black', align='center')

    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel('Predicted probability', fontsize=14)
    ax.set_ylabel('Observed frequency', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True)
    ax.set_xlim(0, 1)

    if len(ci_lowers) > 0 and len(ci_uppers) > 0:
        ylim_lower = -hist_max_height * 1.2
        ylim_upper = max(ci_uppers) * 1.05
    else:
        ylim_lower = -hist_max_height * 1.2
        ylim_upper = 1.0

    ax.set_ylim(ylim_lower, ylim_upper)

    intercept, slope = int_slope_calibration(labels, predictions)

    text_x = 0.45
    text_y = -0.5

    ax.text(
        text_x, text_y,
        f'Intercept = {intercept:.3f}\nSlope = {slope:.3f}',
        fontsize=16, color='black',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    return ax, intercept, slope

def auc_plot(y_true, y_score, title="", ax=None):
    """
    Plot the ROC curve and report the corresponding AUC.

    Parameters
    ----------
    y_true : array-like
        Observed binary outcomes.

    y_score : array-like
        Predicted probabilities or continuous scores.

    title : str, default=""
        Plot title.

    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the ROC curve.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    ax.plot(fpr, tpr, lw=2.5, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(title, fontsize=16)

    ax.legend(loc="lower right", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    ax.grid(alpha=0.3)

