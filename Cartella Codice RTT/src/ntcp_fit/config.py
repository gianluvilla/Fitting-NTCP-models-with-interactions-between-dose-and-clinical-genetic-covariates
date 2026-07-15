"""
Global configuration module for the EUD-NTCP package.

This module stores the default model settings (`DEFAULT_OTHER`), which can
be customized at runtime through `define_settings()` and restored with
`reset_settings()`.

It also automatically builds the coefficient name lists (`COEFF_NAMES_*`)
used to map optimization parameters to the corresponding model coefficients
and to provide consistent parameter labeling in outputs and visualizations.

The `SETTINGS` dictionary contains outcome-specific metadata (e.g., EUD,
PRS, SNP, and outcome variable names) used throughout the package.
"""

import numpy as np

DEFAULT_OTHER = {
    "SNPs_U": [],
    "SNPs_I": [],
    "A_ATM_VALUE": 0.5,
    "A_PRS_VALUE": 0.5,
    "freeze_snps": [],
    "penalize_SNPs_init": True,
    "penalize_SNPs_refinement": True,
    "pATM_threshold": 57.8,
    "OPTIONS": None,
    "OPTIMIZATION_ALGORITHM": "L-BFGS-B",
    "max_bound_gen": np.inf,
    "max_bound_PRS": np.inf
}

OTHER = DEFAULT_OTHER.copy()

OPTIONS = OTHER["OPTIONS"]
OPTIMIZATION_ALGORITHM = OTHER["OPTIMIZATION_ALGORITHM"]
max_bound = OTHER["max_bound_gen"]

COEFF_NAMES_U = []
COEFF_NAMES_I = []
COEFF_NAMES_UI = []

COEFF_NAMES_U_app = []
COEFF_NAMES_I_app = []
COEFF_NAMES_UI_app = []

SETTINGS = {}


def build_settings():
    snps_u = OTHER["SNPs_U"]
    snps_i = OTHER["SNPs_I"]

    coeff_names_U_base = [
        "beta_0_U",
        "beta_EUDb",
        "beta_EUDb_ATM",
        "beta_EUDb_PRS",
        "beta_PRS_U",
        "beta_0_PRS_U",
    ]

    coeff_names_I_base = [
        "beta_0_I",
        "beta_EUDr",
        "beta_EUDr_ATM",
        "beta_EUDr_PRS",
        "beta_PRS_I",
        "beta_0_PRS_I",
    ]

    coeff_names_U = coeff_names_U_base + snps_u
    coeff_names_I = coeff_names_I_base + snps_i
    coeff_names_UI = coeff_names_U_base + coeff_names_I_base + snps_u + snps_i

    coeff_names_U_app = coeff_names_U_base + [c + "(U)" for c in snps_u]
    coeff_names_I_app = coeff_names_I_base + [c + "(I)" for c in snps_i]
    coeff_names_UI_app = (
        coeff_names_U_base
        + coeff_names_I_base
        + [c + "(U)" for c in snps_u]
        + [c + "(I)" for c in snps_i]
    )

    settings = {
        "U": {
            "coeff_names": coeff_names_U,
            "coeff_names_no_prs": coeff_names_U_base[:-1],
            "EUD_name": "EUDb",
            "snps_names": snps_u,
            "PRS_name": "PRS_U",
            "outcome_col": "outcome_U",
            "positive_params": ["beta_EUDb", "beta_PRS_U", "beta_EUDb_ATM"],
        },
        "I": {
            "coeff_names": coeff_names_I,
            "coeff_names_no_prs": coeff_names_I_base[:-1],
            "EUD_name": "EUDr",
            "snps_names": snps_i,
            "PRS_name": "PRS_I",
            "outcome_col": "outcome_I",
            "positive_params": ["beta_EUDr", "beta_PRS_I", "beta_EUDr_ATM"],
        },
        "UI": {
            "coeff_names": coeff_names_UI,
            "positive_params": [
                "beta_EUDb",
                "beta_PRS_U",
                "beta_EUDr",
                "beta_PRS_I",
                "beta_EUDb_ATM",
                "beta_EUDr_ATM",
            ],
            "snps_names": (snps_u, snps_i),
        },
    }

    return {
        "COEFF_NAMES_U": coeff_names_U,
        "COEFF_NAMES_I": coeff_names_I,
        "COEFF_NAMES_UI": coeff_names_UI,
        "COEFF_NAMES_U_app": coeff_names_U_app,
        "COEFF_NAMES_I_app": coeff_names_I_app,
        "COEFF_NAMES_UI_app": coeff_names_UI_app,
        "SETTINGS": settings,
    }


def _update_globals():
    global OPTIONS, OPTIMIZATION_ALGORITHM

    built = build_settings()

    COEFF_NAMES_U[:] = built["COEFF_NAMES_U"]
    COEFF_NAMES_I[:] = built["COEFF_NAMES_I"]
    COEFF_NAMES_UI[:] = built["COEFF_NAMES_UI"]

    COEFF_NAMES_U_app[:] = built["COEFF_NAMES_U_app"]
    COEFF_NAMES_I_app[:] = built["COEFF_NAMES_I_app"]
    COEFF_NAMES_UI_app[:] = built["COEFF_NAMES_UI_app"]

    SETTINGS.clear()
    SETTINGS.update(built["SETTINGS"])

    OPTIONS = OTHER["OPTIONS"]
    OPTIMIZATION_ALGORITHM = OTHER["OPTIMIZATION_ALGORITHM"]


def define_settings(**kwargs):
    """
    Update the global model settings at runtime.

    Accepted keyword arguments are the fields defined in `DEFAULT_OTHER`:

    Parameters
    ----------
    SNPs_U : list of str
        Names of the SNP variables to include in the urinary toxicity model.

    SNPs_I : list of str
        Names of the SNP variables to include in the intestinal toxicity model.

    A_ATM_VALUE : float
        Exponent applied to the EUD term in the EUD × ATM interaction,
        i.e. EUD**A_ATM_VALUE × ATM.

    A_PRS_VALUE : float
        Exponent applied to the EUD term in the EUD × PRS interaction,
        i.e. EUD**A_PRS_VALUE × PRS.

    freeze_snps : list of str
        Names of SNP coefficients to freeze after the initial PRS fitting.
        These coefficients are estimated once during PRS construction and
        then kept fixed throughout the optimization of the complete model.

    penalize_SNPs_init : bool
        If True, SNP coefficients are subjected to the same L1/L2
        regularization as the other coefficients during the initial
        PRS fitting step.

    penalize_SNPs_refinement : bool
        If True, SNP coefficients are subjected to the same L1/L2
        regularization as the other coefficients during optimization
        of the complete model.

    pATM_threshold : float
        Threshold value used to define the ATM-based radiosensitivity
        transformation.

    OPTIONS : dict or None
        Optional dictionary of keyword arguments passed to the SciPy
        optimization routine (e.g. {"maxiter": 10000}).

    OPTIMIZATION_ALGORITHM : str
        Name of the SciPy optimization algorithm (e.g. "L-BFGS-B",
        "trust-constr", "SLSQP").
    
    max_bound_gen: float
        Maximum bound for each parameter. Defaut to np.inf (unbounded)
    
    max_bound_PRS: float
        Maximum bound for beta_PRS_U and beta_PRS_I parameters. Defaut to np.inf (positive and unbounded)

    Examples
    --------
    define_settings(
        SNPs_U=["SNP_1", "SNP_2"],
        SNPs_I=["SNP_3"],
        A_ATM_VALUE=0.5,
        A_PRS_VALUE=0.5,
        freeze_snps=["SNP_2"],
        penalize_SNPs_init=True,
        penalize_SNPs_refinement=False,
        pATM_threshold=57.8,
        OPTIMIZATION_ALGORITHM="L-BFGS-B",
        OPTIONS={"maxiter": 10000},
    )
    """
    allowed_keys = set(OTHER.keys())

    for key in kwargs:
        if key not in allowed_keys:
            raise KeyError(f"Chiave non valida per OTHER: {key}")

    OTHER.update(kwargs)
    _update_globals()


def reset_settings():
    OTHER.clear()
    OTHER.update(DEFAULT_OTHER.copy())
    _update_globals()


_update_globals()