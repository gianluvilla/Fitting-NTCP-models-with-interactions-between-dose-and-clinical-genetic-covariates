from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from .config import COEFF_NAMES_UI_app
from . import config
from .optimization import fit_model_complete
import copy

def bootstrap_sample(data, b, alpha0=100, tol=0.025, max_attempts=200, corr_tol=0.05, dup_min=0.05, dup_max=0.15, SETTINGS_LOCAL=None):
    """ 
    Generate a constrained bootstrap sample preserving outcome incidence, 
    inter-outcome correlation, and a controlled proportion of duplicated patients. 
    """
    if SETTINGS_LOCAL is None:
        SETTINGS_LOCAL = config.SETTINGS

    outcome_I_name = SETTINGS_LOCAL['I']['outcome_col']
    outcome_U_name = SETTINGS_LOCAL['U']['outcome_col']
    n = len(data)
    target_cols = [outcome_U_name, outcome_I_name]
    mu_U = data[outcome_U_name].mean()
    mu_I = data[outcome_I_name].mean()
    var_UI = (mu_U*(1-mu_U)*mu_I*(1-mu_I))
    target_corr = 0.0
    if var_UI > 0:
        target_corr = ((data[outcome_U_name] & data[outcome_I_name]).mean() - mu_U*mu_I) / (var_UI**0.5)
    
    for attempt in range(max_attempts):
        seed = (b * 997 + attempt * 12345) % 2**32
        rng = np.random.default_rng(seed)

        p_U = rng.beta(alpha0*mu_U, alpha0*(1-mu_U))
        n_U = int(rng.binomial(n, p_U))
        p_I = rng.beta(alpha0*mu_I, alpha0*(1-mu_I))
        if p_U in (0.0, 1.0):
            continue
        
        P11_target = p_U * p_I + target_corr * (p_U*(1-p_U)*p_I*(1-p_I))**0.5
        P11_target = max(0.0, min(P11_target, min(p_U, p_I)))

        p_I_given_U1 = P11_target / p_U if p_U > 0 else p_I
        p_I_given_U0 = (p_I - P11_target) / (1 - p_U) if p_U < 1 else p_I
        eps = 1e-6
        p_I_given_U1 = min(max(p_I_given_U1, eps), 1-eps)
        p_I_given_U0 = min(max(p_I_given_U0, eps), 1-eps)

        c11 = int(rng.binomial(n_U, rng.beta(alpha0 * p_I_given_U1, alpha0 * (1 - p_I_given_U1)))) if n_U>0 else 0
        n_U0 = n - n_U
        c01 = int(rng.binomial(n_U0, rng.beta(alpha0 * p_I_given_U0, alpha0 * (1 - p_I_given_U0)))) if n_U0>0 else 0
        c10 = n_U - c11
        c00 = n - (c11 + c10 + c01)
        if min(c00,c10,c01,c11)<0:
            continue

        def sample_group(df, u_val, i_val, n_samples):
            group = df[(df[outcome_U_name]==u_val) & (df[outcome_I_name]==i_val)]
            if len(group)==0:
                idxs = rng.choice(df.index, size=n_samples, replace=True)
            else:
                replace = False if n_samples <= len(group) else True
                idxs = rng.choice(group.index, size=n_samples, replace=replace)
            return df.loc[idxs]

        df_list = [
            sample_group(data, 1,1,c11),
            sample_group(data, 1,0,c10),
            sample_group(data, 0,1,c01),
            sample_group(data, 0,0,c00)
        ]

        bootstrap_data = pd.concat(df_list).sample(frac=1, random_state=rng).reset_index(drop=True)
        means_boot = bootstrap_data[target_cols].mean()
        corr_boot = bootstrap_data[outcome_U_name].corr(bootstrap_data[outcome_I_name])
        n_dup = len(bootstrap_data) - len(bootstrap_data['ID'].unique())
        frac_dup = n_dup / n

        if (all(abs(means_boot - np.array([mu_U, mu_I])) <= tol) and
            abs(corr_boot - target_corr) <= corr_tol and
            dup_min <= frac_dup <= dup_max):
            return bootstrap_data

    return None

def fit_bootstrap(data, N_bootstraps=500, L1=0, L2=0, L_joint=0.5, coverage=0.9):
    """
    Estimate parameter uncertainty using constrained bootstrap resampling.

    Bootstrap datasets are generated while approximately preserving the
    incidence of the urinary and intestinal outcomes, their correlation,
    and a predefined proportion of duplicated patients. The complete
    EUD-NTCP model is then fitted independently to each valid bootstrap
    sample.

    The parameter estimate reported as ``Estimate`` corresponds to the fit
    obtained on the original (complete) dataset. Confidence intervals are
    computed using the percentile bootstrap method, i.e. as the
    ``(1 - coverage)/2`` and ``1 - (1 - coverage)/2`` empirical quantiles
    of the bootstrap estimates.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing the variables required by the EUD-NTCP
        model. An ``ID`` column must be present to evaluate the fraction
        of duplicated patients in each bootstrap sample.

    N_bootstraps : int, default=500
        Number of bootstrap samples to generate and fit.

    L1 : float, default=0
        L1 regularization strength used during model fitting.

    L2 : float, default=0
        L2 regularization strength used during model fitting.

    L_joint : float, default=0.1
        L_joint term regularization strength used during model fitting.

    coverage : float, default=0.9
        Coverage probability of the percentile bootstrap confidence
        intervals (e.g. 0.9 corresponds to 90% confidence intervals).

    Returns
    -------
    dict
        Dictionary containing:

        - ``bootstrap samples``: list of valid bootstrap datasets.
        - ``bootstrap results``: bootstrap distributions of the fitted
          parameters and selected negative log-likelihood components.
    """
    SETTINGS_LOCAL = copy.deepcopy(config.SETTINGS)

    print("Generating bootstrap samples...")
    bootstrap_samples = []

    for seed in range(N_bootstraps):
        bs = bootstrap_sample(
            data,
            seed,
            alpha0=200,
            tol=0.025,
            max_attempts=100000,
            SETTINGS_LOCAL=SETTINGS_LOCAL
        )
        bootstrap_samples.append(bs)

    bootstrap_samples = [b for b in bootstrap_samples if b is not None]

    def fit_UI_try(df, SETTINGS_LOCAL, L1=L1, L2=L2, L_joint=L_joint):
        try:
            import ntcp_fit.optimization as optimization

            optimization.SETTINGS = SETTINGS_LOCAL

            return fit_model_complete(
                df,
                L1=L1,
                L2=L2,
                L_joint=L_joint
            )

        except np.linalg.LinAlgError:
            return []

    print("Fitting bootstrap samples...")
    print("")

    results_bootstrap = Parallel(n_jobs=-1, verbose=5)(
        delayed(fit_UI_try)(df, SETTINGS_LOCAL)
        for df in bootstrap_samples
    )

    results_original = fit_UI_try(data, SETTINGS_LOCAL)

    results_bootstrap = [r for r in results_bootstrap if r != []]

    results_original_estimates = results_original['UI model']['refined_params']

    params_list = COEFF_NAMES_UI_app + ['NLL_UI_tot', 'NLL_I', 'NLL_U']
    bootstrap_values = {param: [] for param in params_list}

    for result in results_bootstrap:
        refined_res_b = result['UI model']['refined_params']
        ui_params = np.array([
            refined_res_b[name]
            for name in refined_res_b.keys()
            if name in params_list
        ])

        for i, param in enumerate(params_list):
            bootstrap_values[param].append(ui_params[i])

    alpha = 1 - coverage
    ci_dict = {}

    for param in params_list:
        vals = np.array(bootstrap_values[param])
        lower = np.quantile(vals, alpha / 2)
        upper = np.quantile(vals, 1 - alpha / 2)
        median = np.median(vals)

        ci_dict[param] = [
            results_original_estimates[param],
            median,
            lower,
            upper
        ]

    print('\n\n')
    print(f"{'Parameter':<20}   {'Estimate [{}% CI]'.format(int(coverage*100)):>25}")
    print("-" * 60)

    for param in ci_dict.keys():
        estimate, median, lower, upper = ci_dict[param]
        print(f"{param:<20}   {estimate:>10.3f} [{lower:.3f}, {upper:.3f}]")

    return {
        'bootstrap samples': bootstrap_samples,
        'bootstrap results': bootstrap_values
    }