import numpy as np
from .config import OTHER, SETTINGS

def sigmoid(x): 
    """Implements the sigmoidal function."""
    x = np.clip(x, -500, 500)
    p = 1 / (1 + np.exp(-x))
    return np.clip(p, 1e-12, 1-(1e-12))

def predict(data, params, setting):
    """
    Compute the predicted toxicity probability for each patient.

    The prediction is obtained from the fitted EUD-NTCP model, including
    the EUD, ATM, PRS, and EUD × (ATM/PRS) interaction terms. The PRS is
    first computed from the selected SNPs and then incorporated into the
    final logistic model.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing the required EUD, ATM, and SNP variables.

    params : array-like
        Model parameter vector. Its length and ordering must match the
        coefficient names defined for the selected setting.

    setting : {"U", "I"}
        Model to use:
        - "U": urinary toxicity model.
        - "I": intestinal toxicity model.

    Returns
    -------
    numpy.ndarray
        Predicted toxicity probabilities for all patients.
    """
    EUD_name = SETTINGS[setting]['EUD_name']
    coeff_names = SETTINGS[setting]['coeff_names']
    snps_names = SETTINGS[setting]['snps_names']

    expected_len = len(coeff_names)
    if len(params) != expected_len:
        raise ValueError(f"Expected params of length {expected_len}, but got length {len(params)}")

    beta_0, beta_EUD, beta_EUD_ATM, beta_EUD_PRS, beta_PRS, beta_0_PRS = params[:6]
    betas_snp = np.array(params[6:len(params)])

    PRS_probs = sigmoid(beta_0_PRS + data.loc[:, snps_names].values @ betas_snp)
    EUDa_ATM = data[EUD_name] ** OTHER['A_ATM_VALUE'] * data['ATM']
    EUDa_PRS = data[EUD_name] ** OTHER['A_PRS_VALUE'] * PRS_probs

    scores = (
        beta_0
        + beta_EUD * data[EUD_name]
        + beta_EUD_ATM * EUDa_ATM
        + beta_EUD_PRS * EUDa_PRS
        + beta_PRS * PRS_probs
    )
    probs = sigmoid(scores)
    return probs

def smooth_L1(x, delta=1e-8):
    """Implements a smoothed version of the L1 penalization."""
    return np.sqrt(x**2 + delta)
