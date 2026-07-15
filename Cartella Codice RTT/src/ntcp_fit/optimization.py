# -------------------------------------------------------------------------
# Optimization pipeline
#
# The fitting procedure consists of three stages:
#
# 1. Initialization
#    - Fit a logistic model to estimate the PRS from the selected SNPs.
#    - Use the estimated PRS to initialize the complete toxicity model.
#
# 2. Refinement
#    - Refine all model parameters by directly minimizing the penalized
#      negative log-likelihood using scipy.optimize.minimize.
#
# 3. Joint optimization
#    - Fit the urinary and intestinal models separately.
#    - Use the resulting estimates as initialization for the combined
#      U+I model, optionally including the joint likelihood penalty.
# -------------------------------------------------------------------------

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from .config import (
    OTHER,
    SETTINGS,
    COEFF_NAMES_U_app,
    COEFF_NAMES_I_app,
    max_bound)

from .utilities import sigmoid, smooth_L1, predict

def fit_logistic_reg(X, y, coeff_names = [], positive_params = [], offset_values = 0.0, L1=0.0, L2=0.0):
    """Fit a logistic regression model with standardized predictors and optional L1/L2 regularization."""
    scaler_obj = StandardScaler()
    X_scaled = scaler_obj.fit_transform(X)
    params, preds = custom_fit(X_scaled, y, coeff_names, positive_params, offset_values, L1, L2)
    beta_scaled = params[1:]
    intercept_scaled = params[0]
    beta_original = beta_scaled / scaler_obj.scale_
    intercept_original = intercept_scaled - np.sum(
        (scaler_obj.mean_ / scaler_obj.scale_) * beta_scaled
    )
    return np.concatenate([[intercept_original], beta_original]), preds

def custom_fit(X, y, coeff_names = [], positive_params = [], offset_values = 0.0, L1=0.0, L2=0.0):
    """Fits a penalized logistic regression model with optional offset values on the scores."""
    n_samples, n_features = X.shape
    def neg_loglik(beta):
        linpred = X @ beta[1:] + beta[0] + offset_values
        loglik = -np.sum(y*linpred - np.log1p(np.exp(linpred)))
        loglik += L1*np.sum(smooth_L1(beta[1:])) + 0.5*L2*np.sum(beta[1:]**2)
        return loglik/n_samples
    beta0 = np.zeros(n_features+1)
    res = minimize(
        neg_loglik,
        beta0,
        method='L-BFGS-B',
        options=None,
        bounds = ([(-np.inf, np.inf)] + OPTIMIZATION_BOUNDS(coeff_names, positive_params)) if len(coeff_names) > 0 else None
    )
    beta_est = res.x
    linpred = X @ beta_est[1:] + beta_est[0] + offset_values
    pred = sigmoid(linpred)
    return beta_est, pred

def fit_logistic_block(data, setting, L1=0.0, L2=0.0):
    """Generate initial model parameters by sequentially fitting the PRS and the logistic toxicity model."""
    EUD_col = SETTINGS[setting]['EUD_name']
    PRS_col = SETTINGS[setting]['PRS_name']
    outcome_col = SETTINGS[setting]['outcome_col']
    coeff_names_no_prs = SETTINGS[setting]['coeff_names_no_prs']
    coeff_names_tot = SETTINGS[setting]['coeff_names']
    snps_names = SETTINGS[setting]['snps_names']
    PRS_name = SETTINGS[setting]['PRS_name']
    
    all_params = [EUD_col, 'EUDa_ATM', 'EUDa_PRS', PRS_col]
    positive_cols = [EUD_col, PRS_col, 'EUDa_ATM']

    df = data.copy()

    X_PRS = df[snps_names].values
    y_prs = df[[outcome_col]].values.ravel()
    
    if OTHER['penalize_SNPs_init']:
        params_prs, PRS_values = fit_logistic_reg(X=X_PRS, y=y_prs, L1=L1, L2=L2)
    else:
        params_prs, PRS_values = fit_logistic_reg(X=X_PRS, y=y_prs)
    
    beta_0_PRS = params_prs[0]
    betas_snp = params_prs[1:]
    df[PRS_name] = PRS_values.copy()

    # Fit Logistic (Initialization)
    df['EUDa_ATM'] = df[EUD_col].values ** OTHER['A_ATM_VALUE'] * df['ATM'].values
    df['EUDa_PRS'] = df[EUD_col].values ** OTHER['A_PRS_VALUE'] * df[PRS_name].values

    X_tot = df[all_params].values
    y_tot = df[[outcome_col]].values.ravel()
    params, _ = fit_logistic_reg(X_tot, y_tot, coeff_names=all_params, positive_params=positive_cols, L1=L1, L2=L2)

    initial_params = {name: np.nan for name in coeff_names_tot}

    initial_params['beta_0_' + setting] = params[0]

    for idx, name in enumerate(coeff_names_no_prs[1:]):
        initial_params[name] = params[idx + 1]
    
    initial_params['beta_0_' + PRS_name] = beta_0_PRS
    for idx, snp_name in enumerate(snps_names):
        initial_params[snp_name] = betas_snp[idx]

    return initial_params

def OPTIMIZATION_BOUNDS(coeff_names, positive_params = []):
    """Create parameter bounds, enforcing positivity constraints for selected coefficients."""
    assert len(coeff_names) >= len(positive_params)
    bounds = []
    for name in coeff_names:
        if name.startswith('beta_PRS'):
            bounds.append((0, OTHER['max_bound_PRS']))
        elif name in positive_params:
            bounds.append((0, max_bound))
        else:
            bounds.append((-max_bound, max_bound))
    return bounds

def neg_loglikelihood(params, data, setting, L1=0.0, L2=0.0, return_all_components = False, precomputed = None):
    """
    Computes the penalized negative log-likelihood of a single toxicity model:
        - urinary model: setting = 'U'
        - intestinal model: setting = 'I'
    """
    EUD_name = SETTINGS[setting]['EUD_name']
    outcome_col = SETTINGS[setting]['outcome_col']
    coeff_names = SETTINGS[setting]['coeff_names']
    snps_names = SETTINGS[setting]['snps_names']

    expected_len = len(coeff_names)
    if len(params) != expected_len:
        raise ValueError(f"Expected params of length {expected_len}, but got length {len(params)}")

    beta_0, beta_EUD, beta_EUD_ATM, beta_EUD_PRS, beta_PRS, beta_0_PRS = params[:6]
    betas_snp = np.array(params[6:len(params)])

    PRS_probs = sigmoid(beta_0_PRS + data.loc[:, snps_names].values @ betas_snp)
    
    if precomputed is None:
        precomputed = compute_std_pen(data, setting)
    
    stds = precomputed['stds']
    EUDa_ATM = precomputed['EUDa_ATM']

    EUDa_PRS = data[EUD_name].values ** OTHER['A_PRS_VALUE'] * PRS_probs

    scores = (
        beta_0
        + beta_EUD * data[EUD_name].values
        + beta_EUD_ATM * EUDa_ATM
        + beta_EUD_PRS * EUDa_PRS
        + beta_PRS * PRS_probs
    )
    probs = sigmoid(scores)

    log_L = np.sum(data[outcome_col].values * np.log(probs) + (1 - data[outcome_col].values) * np.log(1 - probs))

    penalized_params = [beta_EUD, beta_EUD_ATM, beta_EUD_PRS, beta_PRS]

    if OTHER['penalize_SNPs_refinement']:
        for idx in range(len(snps_names)):
            penalized_params.append(betas_snp[idx])

    penalized_params = np.array(penalized_params)
    scaled_params = penalized_params * stds
    penalty_L1 = L1 * np.sum(smooth_L1(scaled_params))
    penalty_L2 = 0.5 * L2 * np.sum(scaled_params ** 2)

    loss = -log_L + penalty_L1 + penalty_L2
    if return_all_components:
        return {'NLL_tot': loss, 'NLL_' + setting + '_no_pen': - log_L, f'pen_L1 (L1 = {L1})': penalty_L1, f'pen_L2 (L2 = {L2})': penalty_L2}
    else:
        return loss / len(data)

def neg_loglikelihood_UI(params, data, L1=0.0, L2=0.0, L_joint = 0.0, return_all_components = False, precomputed = None):
    """Computes the penalized negative log-likelihood of the model for the overall toxicity."""
    coeff_names = SETTINGS['UI']['coeff_names']
    snp_U_names, snp_I_names = SETTINGS['UI']['snps_names']
    EUDb_col = SETTINGS['U']['EUD_name']
    EUDr_col = SETTINGS['I']['EUD_name']
    outcome_U = SETTINGS['U']['outcome_col']
    outcome_I = SETTINGS['I']['outcome_col']

    params_dict = dict(zip(coeff_names, params))
    beta_0_U = params_dict['beta_0_U']
    beta_EUDb = params_dict['beta_EUDb']
    beta_EUDb_ATM = params_dict['beta_EUDb_ATM']
    beta_EUDb_PRS = params_dict['beta_EUDb_PRS']
    beta_PRS_U = params_dict['beta_PRS_U']
    beta_0_PRS_U = params_dict['beta_0_PRS_U']
    beta_0_I = params_dict['beta_0_I']
    beta_EUDr = params_dict['beta_EUDr']
    beta_EUDr_ATM = params_dict['beta_EUDr_ATM']
    beta_EUDr_PRS = params_dict['beta_EUDr_PRS']
    beta_PRS_I = params_dict['beta_PRS_I']
    beta_0_PRS_I = params_dict['beta_0_PRS_I']

    if precomputed is None:
        precomputed = compute_std_pen(data, 'UI')
    
    EUDb_a_ATM = precomputed['EUDb_a_ATM']
    EUDr_a_ATM = precomputed['EUDr_a_ATM']
    stds = precomputed['stds']

    betas_snp_U = np.array([params_dict[n] for n in snp_U_names])
    betas_snp_I = np.array([params_dict[n] for n in snp_I_names])

    PRS_U = sigmoid(beta_0_PRS_U + data.loc[:, snp_U_names].values @ betas_snp_U)
    PRS_I = sigmoid(beta_0_PRS_I + data.loc[:, snp_I_names].values @ betas_snp_I)

    EUDb_vals = data[EUDb_col].values
    EUDr_vals = data[EUDr_col].values
    EUDb_a_PRS = (EUDb_vals ** OTHER['A_PRS_VALUE']) * PRS_U
    EUDr_a_PRS = (EUDr_vals ** OTHER['A_PRS_VALUE']) * PRS_I

    scores_U = (beta_0_U
                + beta_EUDb * EUDb_vals
                + beta_EUDb_ATM * EUDb_a_ATM
                + beta_EUDb_PRS * EUDb_a_PRS
                + beta_PRS_U * PRS_U)
    scores_I = (beta_0_I
                + beta_EUDr * EUDr_vals
                + beta_EUDr_ATM * EUDr_a_ATM
                + beta_EUDr_PRS * EUDr_a_PRS
                + beta_PRS_I * PRS_I)
    probs_U = sigmoid(scores_U)
    probs_I = sigmoid(scores_I)

    probs_joint = 1 - (1 - probs_U) * (1 - probs_I)
    probs_joint = np.clip(probs_joint, 1e-12, 1-1e-12)
    log_L_U = np.sum(data[outcome_U].values * np.log(probs_U) + (1 - data[outcome_U].values) * np.log(1 - probs_U))
    log_L_I = np.sum(data[outcome_I].values * np.log(probs_I) + (1 - data[outcome_I].values) * np.log(1 - probs_I))
    log_L = log_L_U + log_L_I
    log_L_UI = np.sum(data['outcome'].values * np.log(probs_joint) + (1 - data['outcome'].values) * np.log(1 - probs_joint))

    penalized_params = [
        beta_EUDb, beta_EUDb_ATM, beta_EUDb_PRS,
        beta_EUDr, beta_EUDr_ATM, beta_EUDr_PRS,
        beta_PRS_U, beta_PRS_I
    ]

    if OTHER['penalize_SNPs_refinement']:
        for idx in range(len(snp_U_names)):
            penalized_params.append(betas_snp_U[idx])
        for idx in range(len(snp_I_names)):
            penalized_params.append(betas_snp_I[idx])
    
    penalized_params = np.array(penalized_params)

    scaled_params = penalized_params * stds
    penalty_L1 = L1 * np.sum(smooth_L1(scaled_params))
    penalty_L2 = 0.5 * L2 * np.sum(scaled_params ** 2)
    penalty_joint = - L_joint * log_L
    
    loss = - log_L_UI + penalty_joint + penalty_L1 + penalty_L2

    if return_all_components:
        return {'NLL_UI_tot': loss, 
                'NLL_UI': - log_L_UI,
                'NLL_U': - log_L_U,
                'NLL_I': - log_L_I,
                'NLL_UI_no_pen': - log_L_UI + penalty_joint, 
                f'pen_L1 (L1 = {L1})': penalty_L1, f'pen_L2 (L2 = {L2})': penalty_L2, f'pen_J (LJ = {L_joint})': penalty_joint}
    else:
        return loss / len(data)

def compute_std_pen(data, setting):
    """Precompute feature standard deviations used to scale L1/L2 regularization terms."""
    if setting in ('U', 'I'):
        EUD_name = SETTINGS[setting]['EUD_name']
        snps_names = SETTINGS[setting]['snps_names']
        outcome_col = SETTINGS[setting]['outcome_col']
        p = np.mean(data[outcome_col].values)
        EUDa_ATM = data[EUD_name].values ** OTHER['A_ATM_VALUE'] * data['ATM'].values

        _eud = data[EUD_name].values.std(ddof=0)
        std_eud = _eud if _eud != 0 else 1.0

        _eud_atm = EUDa_ATM.std(ddof=0)
        std_eud_atm = _eud_atm if _eud_atm != 0 else 1.0

        _std_prs = np.sqrt(p * (1 - p))
        std_prs = _std_prs if _std_prs != 0 else 1.0

        _std_eud_prs = std_eud * std_prs
        std_eud_prs = _std_eud_prs if _std_eud_prs != 0 else 1.0

        stds = [
            std_eud,
            std_eud_atm,
            std_eud_prs,
            std_prs
        ]
        if OTHER['penalize_SNPs_refinement']:
            for snp in snps_names:
                std_snp = data[snp].values.std(ddof=0)
                if std_snp != 0:
                    stds.append(std_snp)
                else:
                    stds.append(1.0)
        return {'stds': np.array(stds), 'EUDa_ATM': EUDa_ATM}
    else:
        snp_U_names, snp_I_names = SETTINGS['UI']['snps_names']
        EUDb_col = SETTINGS['U']['EUD_name']
        EUDr_col = SETTINGS['I']['EUD_name']
        outcome_U = SETTINGS['U']['outcome_col']
        outcome_I = SETTINGS['I']['outcome_col']
        EUDb_vals = data[EUDb_col].values
        EUDr_vals = data[EUDr_col].values
        EUDb_a_ATM = (EUDb_vals ** OTHER['A_ATM_VALUE']) * data['ATM'].values
        EUDr_a_ATM = (EUDr_vals ** OTHER['A_ATM_VALUE']) * data['ATM'].values
        p_I = np.mean(data[outcome_I].values)
        p_U = np.mean(data[outcome_U].values)
        _std_prs_u = np.sqrt(p_U * (1-p_U))
        _std_prs_i = np.sqrt(p_I * (1-p_I))
        _eudb = EUDb_vals.std(ddof=0)
        _eudb_atm = EUDb_a_ATM.std(ddof=0)
        _eudb_prs = _eudb * _std_prs_u
        _eudr = EUDr_vals.std(ddof=0)
        _eudr_atm = EUDr_a_ATM.std(ddof=0)
        _eudr_prs = _eudr * _std_prs_i

        stds = [
            _eudb if _eudb != 0 else 1.0,
            _eudb_atm if _eudb_atm != 0 else 1.0,
            _eudb_prs if _eudb_prs != 0 else 1.0,
            _eudr if _eudr != 0 else 1.0,
            _eudr_atm if _eudr_atm != 0 else 1.0,
            _eudr_prs if _eudr_prs != 0 else 1.0,
            _std_prs_u if _std_prs_u != 0 else 1.0,
            _std_prs_i if _std_prs_i != 0 else 1.0
        ]
        if OTHER['penalize_SNPs_refinement']:
            for snp in snp_U_names:
                _std_snp = data[snp].std(ddof=0)
                stds.append(_std_snp if _std_snp != 0 else 1.0)
            for snp in snp_I_names:
                _std_snp = data[snp].std(ddof=0)
                stds.append(_std_snp if _std_snp != 0 else 1.0)
        
        return {'stds': np.array(stds), 'EUDb_a_ATM': EUDb_a_ATM, 'EUDr_a_ATM': EUDr_a_ATM}

def refine_single_point(params_dict, data, setting, L1=0.0, L2=0.0, L_joint = 0.0):
    """Refine an initial parameter estimate by direct optimization of the penalized likelihood."""
    coeff_names = SETTINGS[setting]['coeff_names']
    positive_params = SETTINGS[setting]['positive_params']
    settings_freeze_SNPs = OTHER['freeze_snps']

    if list(params_dict.keys()) == coeff_names:
        x0 = np.array(list(params_dict.values()))
    else:
        raise ValueError
    
    precomputed = compute_std_pen(data, setting)

    if setting == 'UI':
        nll_func = neg_loglikelihood_UI
        args = (data, L1, L2, L_joint, False, precomputed)
        nll_start = nll_func(x0, data, L1, L2, L_joint, False, precomputed)
    elif setting in ('U', 'I'):
        nll_func = neg_loglikelihood
        args = (data, setting, L1, L2, False, precomputed)
        nll_start = nll_func(x0, data, setting, L1, L2, False, precomputed)
    
    bounds = OPTIMIZATION_BOUNDS(coeff_names, positive_params)

    if setting in settings_freeze_SNPs:
        if setting in ('U', 'I'):
            snp_names = [f'beta_0_PRS_{setting}'] + SETTINGS[setting]['snps_names']
        else:
            snp_U_names, snp_I_names = SETTINGS[setting]['snps_names']
            snp_names = ['beta_0_PRS_U'] + snp_U_names + [f'beta_0_PRS_I'] + snp_I_names
        
        for i, name in enumerate(coeff_names):
            if name in snp_names:
                bounds[i] = (x0[i], x0[i])

    result = minimize(
        nll_func,
        x0=x0,
        args=args,
        method=OTHER['OPTIMIZATION_ALGORITHM'],
        bounds=bounds,
        options=OTHER['OPTIONS']
    )

    assert len(result.x) == len(coeff_names)

    nll_final = result.fun
    if (not result.success) or (not np.isfinite(nll_final)) or (nll_final > nll_start):
        # x_out = x0
        # print(f'OTTIMIZZAZIONE {setting} FALLITA')
        # print(f'Success: {bool(result.success)}')
        # print(f'NLL final: {nll_final:.3f}, NLL_start: {nll_start:.3f}')
        success_flag = False
    else:
        success_flag = True
    x_out = result.x
    res_out = {coeff_names[idx]: x_out[idx] for idx in range(len(x_out))}
    return {'params': res_out, 'opt_success': success_flag}

def fit_model_sing(data, setting, L1=0.0, L2=0.0):    
    """Fit a single toxicity model by logistic initialization followed by likelihood refinement."""
    # Logistic Initialization 
    initial_params = fit_logistic_block(data, setting, L1 = L1, L2 = L2)
        
    # Refinement
    results_refined = refine_single_point(initial_params, data, setting, L1 = L1, L2 = L2)

    refined_params = results_refined['params']
    opt_success = results_refined['opt_success']

    # Compute likelihoods (with and without penalization)
    initial_params_with_loss = initial_params.copy()
    initial_params_array = np.array(list(initial_params.values()))
    initial_params_with_loss.update(neg_loglikelihood(initial_params_array, data, setting, L1 = L1, L2 = L2, return_all_components=True))

    refined_params_with_loss = refined_params.copy()
    refined_params_array = np.array(list(refined_params.values()))
    refined_params_with_loss.update(neg_loglikelihood(refined_params_array, data, setting, L1 = L1, L2 = L2, return_all_components=True))
    refined_params_with_loss['opt_success'] = opt_success

    return {'refined_params': refined_params_with_loss, 'initial_params': initial_params_with_loss}

def fit_model_complete(data, L1=0.0, L2=0.0, L_joint=0.0):
    """Fit the urinary, intestinal, and combined models in a sequential optimization pipeline."""
    coeff_names_U = SETTINGS['U']['coeff_names']
    coeff_names_I = SETTINGS['I']['coeff_names']
    coeff_names_combined = SETTINGS['UI']['coeff_names']
    snp_U_names, snp_I_names = SETTINGS['UI']['snps_names']

    # Fit partial models
    model_U = fit_model_sing(data, 'U', L1=L1, L2=L2)
    model_I = fit_model_sing(data, 'I', L1=L1, L2=L2)

    # Combine parameters
    params_U = model_U['refined_params']
    params_I = model_I['refined_params']
    initial_combined = {}
    for name in coeff_names_combined:
        if name in coeff_names_U:
            initial_combined[name] = params_U[name]
        elif name in coeff_names_I:
            initial_combined[name] = params_I[name]
    
    results_refined_combined = refine_single_point(initial_combined, data, 'UI', L1=L1, L2=L2, L_joint=L_joint)
    
    refined_combined = results_refined_combined['params']
    opt_success = results_refined_combined['opt_success']

    def rename_combined(initial_dict):
        ren_dict = {}
        for k, v in initial_dict.items():
            if k in snp_U_names:
                new_key = k + "(U)"
            elif k in snp_I_names:
                new_key = k + "(I)"
            else:
                new_key = k
            ren_dict[new_key] = v
        return ren_dict

    initial_combined = rename_combined(initial_combined).copy()    
    initial_combined_array = np.array(list(initial_combined.values()))
    initial_combined_with_loss = initial_combined.copy()
    initial_combined_with_loss.update(neg_loglikelihood_UI(initial_combined_array, data, L1 = L1, L2 = L2, L_joint=L_joint, return_all_components=True))

    refined_combined = rename_combined(refined_combined).copy()
    refined_combined_array = np.array(list(refined_combined.values()))
    refined_combined_with_loss = refined_combined.copy()
    refined_combined_with_loss.update(neg_loglikelihood_UI(refined_combined_array, data, L1 = L1, L2 = L2, L_joint=L_joint, return_all_components=True))

    refined_combined_with_loss['opt_success'] = opt_success

    return {'UI model': {'refined_params': refined_combined_with_loss, 'initial_params': initial_combined_with_loss},
            'U model (partial)': model_U,
            'I model (partial)': model_I}

def fit_model(data, L1=0, L2=0, L_joint=0.5):
    """
    Fit the complete EUD-NTCP model and compute patient-specific predictions.

    The fitting procedure consists of three stages:

    1. A logistic regression is fitted to estimate the polygenic risk score (PRS)
       from the selected SNPs.
    2. The PRS is used to initialize the toxicity model, which is then refined by
       minimizing the penalized negative log-likelihood.
    3. Separate urinary and intestinal models are combined into a joint model,
    including a joint likelihood penalty controlled by `L_joint`.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing the EUD, ATM, SNP, and outcome variables.

    L1 : float, default=0
        L1 regularization strength.

    L2 : float, default=0
        L2 regularization strength.

    L_joint : float, default=0.5
        Weight of the joint likelihood penalty used during optimization of the
        combined urinary-intestinal model.

        The default value (0.5) helps preserve the calibration and predictive
        performance of the individual urinary and intestinal models after the
        joint refinement, while generally maintaining similar performance on the
        combined endpoint.

    Returns
    -------
    dict
        Dictionary containing:

        - ``results`` : complete fitting results, including initial and refined
          parameter estimates for all models.
        - ``params_U_model`` : fitted parameter vector for the urinary model.
        - ``params_I_model`` : fitted parameter vector for the intestinal model.
        - ``predictions_U`` : predicted urinary toxicity probabilities.
        - ``predictions_I`` : predicted intestinal toxicity probabilities.
        - ``predictions_overall`` : predicted probability of developing at least
          one toxicity.
        - ``PRS_U`` : estimated urinary polygenic risk scores.
        - ``PRS_I`` : estimated intestinal polygenic risk scores.
    """
    results_full = fit_model_complete(
        data,
        L1=L1,
        L2=L2,
        L_joint=L_joint
    )

    refined_res = results_full['UI model']['refined_params']
    model_UI = results_full['UI model']['refined_params']

    keys = model_UI.keys()

    name_w = max(len(k) for k in keys) + 2
    val_w = 14

    header = f"{'Parameter':<{name_w}}{'Estimate':>{val_w}}"
    print(header)
    print('-' * len(header))

    for k, value in model_UI.items():
        if k == "NLL_UI_tot":
            print()

        if k == "opt_success":
            print(f"{k:<{name_w}}{str(bool(value)):>{val_w}}")
        else:
            print(f"{k:<{name_w}}{value:>{val_w}.5f}")

    params_U = np.array([
        refined_res[name]
        for name in COEFF_NAMES_U_app
    ])

    params_I = np.array([
        refined_res[name]
        for name in COEFF_NAMES_I_app
    ])

    params_U_prs = np.array([
        refined_res[name]
        for name in (
            ['beta_0_PRS_U']
            + [s + '(U)' for s in OTHER['SNPs_U']]
        )
    ])

    params_I_prs = np.array([
        refined_res[name]
        for name in (
            ['beta_0_PRS_I']
            + [s + '(I)' for s in OTHER['SNPs_I']]
        )
    ])

    PRS_U_probs = sigmoid(
        params_U_prs[0]
        + data.loc[:, OTHER['SNPs_U']].values @ params_U_prs[1:]
    )

    PRS_I_probs = sigmoid(
        params_I_prs[0]
        + data.loc[:, OTHER['SNPs_I']].values @ params_I_prs[1:]
    )

    pred_U = predict(data, params_U, 'U')
    pred_I = predict(data, params_I, 'I')

    pred_overall = 1 - (1 - pred_U) * (1 - pred_I)

    return {
        "results": results_full,
        "params_U_model": params_U,
        "params_I_model": params_I,
        "predictions_I": pred_I,
        "predictions_U": pred_U,
        "predictions_overall": pred_overall,
        "PRS_U": PRS_U_probs,
        "PRS_I": PRS_I_probs
    }
