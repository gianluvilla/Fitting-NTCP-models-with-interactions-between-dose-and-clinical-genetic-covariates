"""
Utilities for generating a binary toxicity outcome consistent with a
predefined logistic model.

General logic
-------------
The outcome-generation procedure follows these steps:

1. A model score without intercept is calculated for every patient.

2. The intercept is calibrated so that the average predicted probability
   matches the desired toxicity incidence.

3. The model probabilities are obtained through the logistic function:

       p_i = sigmoid(beta_0 + score_i)

4. A binary outcome vector is generated while satisfying two conditions:

   - its observed incidence must remain close to the requested incidence;
   - the assumed model parameters should be approximately a stationary point
     of the log-likelihood.

For a logistic model, stationarity requires:

       X.T @ (y - p) ≈ 0

where:

    X = matrix containing the derivatives of the linear predictor with
        respect to the fitted parameters;

    y = generated binary outcome;

    p = probabilities predicted by the assumed model.

The search therefore minimizes the squared norm of the likelihood gradient:

       ||X.T @ (y - p)||²

while constraining the number of observed events to remain within the
specified incidence tolerance.

Two generation methods are available:

- "tilted_search":
    starts from multiple probability-weighted outcome vectors and improves
    each one through local flips and swaps;

- "random":
    generates many candidate outcome vectors and retains the one with the
    smallest stationarity objective.
"""

import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from joblib import Parallel, delayed

def calibrate_intercept(score_no_intercept, target_incidence):
    """
    Calibrate the logistic-model intercept to obtain the target incidence.

    The function finds beta_0 such that the mean predicted probability:

        mean(sigmoid(beta_0 + score_no_intercept))

    is equal to `target_incidence`.

    Parameters
    ----------
    score_no_intercept : array-like
        Linear predictor values excluding the intercept.

    target_incidence : float
        Desired mean probability of the outcome.

    Returns
    -------
    float
        Calibrated intercept beta_0.
    """
    score_no_intercept = np.asarray(score_no_intercept, dtype=float)

    def f(beta_0):
        return expit(beta_0 + score_no_intercept).mean() - target_incidence

    return brentq(f, -100, 100)


def make_stationarity_matrix(EUD, PRS, radiosensitivity, SNPs=None, beta_PRS=None, beta_EUD_PRS=None, beta_radiosensitivity=None, a=0.5,
    fit_prs_params=True, fit_a=False, include_intercept=True):
    """
    Build the matrix used to impose likelihood stationarity.

    Each column contains the derivative of the model linear predictor with
    respect to one fitted parameter. Consequently:

        X.T @ (y - p)

    corresponds to the likelihood gradient evaluated at the assumed parameter
    values.

    The base columns refer to the coefficients of:

        intercept
        EUD
        PRS
        PRS * EUD**a
        radiosensitivity * EUD**a
    """
    EUD = np.asarray(EUD, dtype=float)
    PRS = np.asarray(PRS, dtype=float)
    radiosensitivity = np.asarray(radiosensitivity, dtype=float)

    if len(EUD) != len(PRS) or len(EUD) != len(radiosensitivity):
        raise ValueError("EUD, PRS and radiosensitivity must have the same length.")

    n = len(EUD)
    EUDa = EUD ** a

    cols = []

    if include_intercept:
        cols.append(np.ones(n))

    cols.extend([
        EUD,
        PRS,
        PRS * EUDa,
        radiosensitivity * EUDa,
    ])

    matrices = [np.column_stack(cols)]

    if fit_prs_params:
        if SNPs is None:
            raise ValueError("SNPs must be provided if fit_prs_params=True.")
        if beta_PRS is None or beta_EUD_PRS is None:
            raise ValueError("beta_PRS and beta_EUD_PRS are required if fit_prs_params=True.")

        SNPs = np.asarray(SNPs, dtype=float)

        if SNPs.ndim == 1:
            SNPs = SNPs[:, None]

        if SNPs.shape[0] != n:
            raise ValueError("SNPs must have the same number of rows as EUD.")

        dPRS = PRS * (1.0 - PRS)
        d_eta_d_PRS = beta_PRS + beta_EUD_PRS * EUDa
        common_prs = d_eta_d_PRS * dPRS

        X_prs = np.column_stack([
            common_prs,
            SNPs * common_prs[:, None],
        ])

        matrices.append(X_prs)

    if fit_a:
        if beta_EUD_PRS is None or beta_radiosensitivity is None:
            raise ValueError(
                "beta_EUD_PRS and beta_radiosensitivity are required if fit_a=True."
            )

        EUD_safe = np.clip(EUD, 1e-12, None)

        d_eta_d_a = (
            beta_EUD_PRS * PRS * EUDa * np.log(EUD_safe)
            + beta_radiosensitivity * radiosensitivity * EUDa * np.log(EUD_safe)
        )

        matrices.append(d_eta_d_a.reshape(-1, 1))

    return np.column_stack(matrices)


def _validate_design(X, n):
    """
    Validate and standardize a stationarity design matrix.

    A one-dimensional input is converted into a single-column matrix, and the
    function verifies that the number of rows matches the number of patients.

    Parameters
    ----------
    X : array-like
        Candidate design or stationarity matrix.

    n : int
        Expected number of rows.

    Returns
    -------
    numpy.ndarray
        Validated two-dimensional matrix.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X[:, None]

    if X.shape[0] != n:
        raise ValueError("X must have the same number of rows as p.")

    return X


def _stationarity_objective(y, p, X):
    """
    Evaluate how far the assumed parameters are from likelihood stationarity.

    For logistic regression, the likelihood gradient is:

        gradient = X.T @ (y - p)

    The objective is the squared Euclidean norm of this gradient. A value close
    to zero means that the assumed parameter vector is approximately a
    stationary point of the likelihood for the generated outcome.
    """
    grad = X.T @ (y - p)
    return np.sum(grad**2), grad


def _sample_initial_outcome(p, rng, min_events, max_events):
    """
    Generate an initial binary outcome using model probabilities as priorities.

    The number of events is sampled within the permitted incidence range.
    Patients with higher predicted probabilities are more likely to be selected
    as events, although a random component preserves variability between
    different starting solutions.

    Parameters
    ----------
    p : array-like
        Model-predicted event probabilities.

    rng : numpy.random.Generator
        Random-number generator.

    min_events : int
        Minimum permitted number of events.

    max_events : int
        Maximum permitted number of events.

    Returns
    -------
    numpy.ndarray
        Initial binary outcome vector.
    """    
    n = len(p)
    p_safe = np.clip(p, 1e-12, 1.0)

    target_events = rng.integers(min_events, max_events + 1)

    priority = rng.random(n) / p_safe

    y = np.zeros(n, dtype=int)
    y[np.argsort(priority)[:target_events]] = 1

    return y


def improve_stationarity_with_incidence_tolerance(y, p, X, rng, incidence_rate, incidence_tolerance=0.05, n_iter=200000, flip_probability=0.7,):
    """
    Improve an outcome vector through local flips and event/non-event swaps.

    At each iteration, the algorithm proposes either:

    - a flip of one outcome value, which may change the total event count;
    - a swap between one event and one non-event, which preserves incidence.

    A proposal is accepted only when it reduces the squared likelihood-gradient
    norm. Single flips are also required to preserve the allowed incidence
    interval.
    """
    y = np.asarray(y, dtype=int).copy()
    p = np.asarray(p, dtype=float)
    X = _validate_design(X, len(p))

    n = len(y)

    min_events = int(np.floor((incidence_rate - incidence_tolerance) * n))
    max_events = int(np.ceil((incidence_rate + incidence_tolerance) * n))

    min_events = int(np.clip(min_events, 0, n))
    max_events = int(np.clip(max_events, 0, n))

    grad = X.T @ (y - p)
    best_obj = np.sum(grad**2)

    for _ in range(n_iter):
        n_events = int(y.sum())

        if rng.random() < flip_probability:
            idx = rng.integers(n)

            delta_y = 1 - 2 * y[idx]
            new_events = n_events + delta_y

            if new_events < min_events or new_events > max_events:
                continue

            grad_new = grad + delta_y * X[idx]
            obj_new = np.sum(grad_new**2)

            if obj_new < best_obj:
                y[idx] = 1 - y[idx]
                grad = grad_new
                best_obj = obj_new

        else:
            one_idx = np.where(y == 1)[0]
            zero_idx = np.where(y == 0)[0]

            if len(one_idx) == 0 or len(zero_idx) == 0:
                continue

            idx1 = rng.choice(one_idx)
            idx0 = rng.choice(zero_idx)

            grad_new = grad - X[idx1] + X[idx0]
            obj_new = np.sum(grad_new**2)

            if obj_new < best_obj:
                y[idx1] = 0
                y[idx0] = 1
                grad = grad_new
                best_obj = obj_new

    return y, {
        "gradient": grad,
        "gradient_norm": np.linalg.norm(grad),
        "max_abs_gradient": np.max(np.abs(grad)),
        "objective": best_obj,
        "observed_incidence": y.mean(),
        "observed_events": int(y.sum()),
        "min_allowed_events": min_events,
        "max_allowed_events": max_events,
    }


def generate_stationary_outcome_by_tilted_search( p, X, rng, incidence_rate, incidence_tolerance=0.05, n_starts=200, n_iter_per_start=50000, flip_probability=0.7, n_jobs=-1,):
    """
    Generate an approximately stationary outcome using multiple local searches.

    Several probability-weighted initial outcomes are generated independently.
    Each initial solution is improved through flips and swaps, and the solution
    with the smallest squared likelihood-gradient norm is retained.

    Independent restarts are executed in parallel to explore different regions
    of the binary outcome space and reduce dependence on a single local optimum.
    """

    p = np.asarray(p, dtype=float)
    X = _validate_design(X, len(p))

    n = len(p)

    min_events = int(np.floor((incidence_rate - incidence_tolerance) * n))
    max_events = int(np.ceil((incidence_rate + incidence_tolerance) * n))

    min_events = int(np.clip(min_events, 0, n))
    max_events = int(np.clip(max_events, 0, n))

    if min_events > max_events:
        raise ValueError("Invalid incidence tolerance.")

    seeds = rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=n_starts,
        dtype=np.uint32,
    )

    def _single_restart(seed):
        local_rng = np.random.default_rng(int(seed))

        y0 = _sample_initial_outcome(
            p=p,
            rng=local_rng,
            min_events=min_events,
            max_events=max_events,
        )

        y, info = improve_stationarity_with_incidence_tolerance(
            y=y0,
            p=p,
            X=X,
            rng=local_rng,
            incidence_rate=incidence_rate,
            incidence_tolerance=incidence_tolerance,
            n_iter=n_iter_per_start,
            flip_probability=flip_probability,
        )

        return y, info

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
    )(
        delayed(_single_restart)(seed)
        for seed in seeds
    )

    best_y = None
    best_info = None
    best_obj = np.inf

    for y, info in results:
        if info["objective"] < best_obj:
            best_obj = info["objective"]
            best_y = y.copy()
            best_info = info

    return best_y, best_info

def generate_outcome(
    scores_no_intercept,
    incidence_rate,
    rng,
    design_matrix=None,
    include_score=False,
    generation_method="tilted_search",
    incidence_tolerance=0.05,
    n_candidates=50000,
    n_starts=200,
    n_iter_per_start=50000,
    flip_probability=0.7,
    return_info=False,
):
    """
    Generate a binary outcome from an assumed logistic toxicity model.

    The function first calibrates the model intercept so that the mean predicted
    probability equals the requested incidence. It then generates a binary
    outcome whose observed incidence is close to the target and for which the
    assumed model parameters are approximately a stationary point of the
    likelihood.

    Parameters
    ----------
    scores_no_intercept : array-like
        Model linear predictor excluding the intercept.

    incidence_rate : float
        Desired event incidence.

    rng : numpy.random.Generator
        Random-number generator.

    design_matrix : array-like, optional
        Matrix containing derivatives of the model predictor with respect to
        parameters for which stationarity should be imposed.

    include_score : bool, default=False
        Whether to impose stationarity with respect to a global coefficient
        multiplying `scores_no_intercept`.

    generation_method : {"tilted_search", "random"}, default="tilted_search"
        Method used to search for the binary outcome.

    incidence_tolerance : float, default=0.05
        Maximum permitted absolute deviation from the target incidence.

    n_candidates : int, default=50000
        Number of candidates evaluated by the random-search method.

    n_starts : int, default=200
        Number of independent restarts used by the tilted-search method.

    n_iter_per_start : int, default=50000
        Number of local-search iterations performed for each restart.

    flip_probability : float, default=0.7
        Probability of proposing a single outcome flip rather than a swap.

    return_info : bool, default=False
        Whether to return generation diagnostics in addition to the outcome.

    Returns
    -------
    outcome : numpy.ndarray
        Generated binary outcome vector.

    info : dict, optional
        Returned only when `return_info=True`. Contains the calibrated
        intercept, predicted probabilities, incidence information and
        stationarity diagnostics.
    """
    scores_no_intercept = np.asarray(scores_no_intercept, dtype=float)
    n = len(scores_no_intercept)

    beta_0 = calibrate_intercept(
        score_no_intercept=scores_no_intercept,
        target_incidence=incidence_rate,
    )

    eta = beta_0 + scores_no_intercept
    p = expit(eta)

    matrices = []

    if include_score:
        matrices.append(scores_no_intercept.reshape(-1, 1))

    if design_matrix is not None:
        X = _validate_design(design_matrix, n)
        matrices.append(X)

    if len(matrices) == 0:
        score_matrix = np.ones((n, 1))
    else:
        score_matrix = np.column_stack(matrices)

    if generation_method == "tilted_search":
        outcome, search_info = generate_stationary_outcome_by_tilted_search(
            p=p,
            X=score_matrix,
            rng=rng,
            incidence_rate=incidence_rate,
            incidence_tolerance=incidence_tolerance,
            n_starts=n_starts,
            n_iter_per_start=n_iter_per_start,
            flip_probability=flip_probability,
        )

        balance_objective = search_info["objective"]

    elif generation_method == "random":
        best_y = None
        best_obj = np.inf
        best_grad = None

        min_events = int(np.floor((incidence_rate - incidence_tolerance) * n))
        max_events = int(np.ceil((incidence_rate + incidence_tolerance) * n))

        min_events = int(np.clip(min_events, 0, n))
        max_events = int(np.clip(max_events, 0, n))

        for _ in range(n_candidates):
            y = _sample_initial_outcome(
                p=p,
                rng=rng,
                min_events=min_events,
                max_events=max_events,
            )

            obj, grad = _stationarity_objective(y, p, score_matrix)

            if obj < best_obj:
                best_y = y.copy()
                best_obj = obj
                best_grad = grad.copy()

        outcome = best_y
        balance_objective = best_obj

        search_info = {
            "gradient": best_grad,
            "gradient_norm": np.linalg.norm(best_grad),
            "max_abs_gradient": np.max(np.abs(best_grad)),
            "objective": best_obj,
            "observed_incidence": outcome.mean(),
            "observed_events": int(outcome.sum()),
        }

    else:
        raise ValueError("generation_method must be 'tilted_search' or 'random'.")

    if return_info:
        info = {
            "beta_0": beta_0,
            "eta": eta,
            "p": p,
            "expected_incidence": p.mean(),
            "observed_incidence": outcome.mean(),
            "target_events": int(round(p.sum())),
            "observed_events": int(outcome.sum()),
            "incidence_tolerance": incidence_tolerance,
            "balance_objective": balance_objective,
            "search_info": search_info,
        }

        return outcome, info

    return outcome

def check_stationarity(outcome, p, stationarity_matrix):
    """
    Check likelihood stationarity for an already generated outcome.

    The function computes:

        gradient = X.T @ (outcome - p)

    and returns different summaries of its magnitude. Smaller values indicate
    that the supplied parameter values are closer to a stationary point of the
    likelihood.

    Parameters
    ----------
    outcome : array-like
        Observed or generated binary outcome.

    p : array-like
        Predicted probabilities evaluated at the parameter values being tested.

    stationarity_matrix : array-like
        Matrix containing derivatives of the linear predictor with respect to
        the parameters being tested.

    Returns
    -------
    dict
        Dictionary containing the complete gradient, its Euclidean norm, its
        largest absolute component and the sum of squared components.
    """
    outcome = np.asarray(outcome, dtype=float)
    p = np.asarray(p, dtype=float)
    X = _validate_design(stationarity_matrix, len(p))

    grad = X.T @ (outcome - p)

    return {
        "gradient": grad,
        "gradient_norm": np.linalg.norm(grad),
        "max_abs_gradient": np.max(np.abs(grad)),
        "objective": np.sum(grad**2),
    }