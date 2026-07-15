"""
Bootstrap-based ranking of SNP predictors using backward elimination.

This code ranks SNP variables according to their contribution to the
prediction of a binary toxicity outcome.

For each bootstrap iteration, subjects with and without toxicity 
are resampled separately with replacement. 
The SNPs are then ranked through a backward elimination procedure based on
logistic regression. At each step, each remaining SNP is removed in turn,
and the log-likelihood of the corresponding reduced model is calculated.
The SNP whose removal produces the highest log-likelihood is excluded,
because it is considered the least informative among the remaining
predictors.

The exclusion order is finally reversed so that rank 1 corresponds to the
SNP retained for the longest time and therefore considered the most
informative predictor.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

def run_backward_ranking(data, SNPs_list, endpoint):
    """
    Rank SNP predictors using backward logistic-regression elimination.

    At each iteration, every remaining SNP is removed separately and a
    logistic regression model is fitted using the other remaining SNPs.

    The log-likelihood of each reduced model is recorded. The SNP whose
    removal produces the highest log-likelihood is considered the least
    informative predictor at that step and is excluded.

    The procedure continues until all SNPs have been removed. The exclusion
    order is then reversed, so that the SNP excluded last receives rank 1.

    Parameters
    ----------
    data : dict
        Dictionary containing:

        - `data['SNPs']`: pandas DataFrame containing the SNP predictors.
        - `data[endpoint]`: pandas Series containing the binary outcome.

    SNPs_list : list of str
        Names of the SNP predictors to rank.

    endpoint : str
        Key in `data` identifying the binary outcome.

    Returns
    -------
    list of tuple
        Ordered list of `(SNP_name, rank)` pairs. Rank 1 corresponds to the
        SNP retained until the final backward-elimination step.

    Notes
    -----
    If a reduced logistic regression model cannot be fitted because of a
    linear algebra error, its log-likelihood is assigned a value of negative
    infinity. This prevents that candidate model from being selected as the
    best reduced model at the current step.
    """
    y = data[endpoint]

    remaining = SNPs_list.copy()
    excluded = []

    while len(remaining) > 0:
        ll_candidates = {}

        for snp in remaining:
            reduced = [s for s in remaining if s != snp]

            if len(reduced) == 0:
                X_red = pd.DataFrame(index=y.index)
            else:
                X_red = data['SNPs'][reduced].copy()

            X_red = sm.add_constant(X_red)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_red = sm.Logit(y, X_red).fit(disp=False)
                ll_candidates[snp] = model_red.llf
            except np.linalg.LinAlgError:
                ll_candidates[snp] = -np.inf

        worst_snp = max(ll_candidates, key=ll_candidates.get)
        excluded.append(worst_snp)
        remaining.remove(worst_snp)

    ranking_snps = excluded[::-1]

    return [(snp, i + 1) for i, snp in enumerate(ranking_snps)]


def run_bootstrap_single(data, SNPs_list, outcome_key, b):
    """
    Perform a single stratified bootstrap iteration and rank the SNP predictors using 'run_backward_ranking'.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing the SNP variables and the binary outcome.

    SNPs_list : list of str
        Names of the columns containing the SNP predictors to be ranked.

    outcome_key : str
        Name of the column containing the binary outcome, encoded as 0 and 1.

    b : int
        Bootstrap iteration index. It is used as the random seed so that
        the bootstrap sample is reproducible.

    Returns
    -------
    list of tuple
        List of `(SNP_name, rank)` pairs returned by
        `run_backward_ranking`. Rank 1 identifies the SNP considered the
        most informative.
    """
    y_all = data[outcome_key].values
    idx0 = np.where(y_all == 0)[0]
    idx1 = np.where(y_all == 1)[0]

    np.random.seed(b)
    sample0 = np.random.choice(idx0, size=len(idx0), replace=True)
    sample1 = np.random.choice(idx1, size=len(idx1), replace=True)
    sample_idx = np.concatenate([sample0, sample1])

    data_boot = {
        'SNPs': data[SNPs_list].iloc[sample_idx].reset_index(drop=True),
        'outcome': data[outcome_key].iloc[sample_idx].reset_index(drop=True)
    }
    
    ranking_b = run_backward_ranking(
        data_boot,
        SNPs_list,
        endpoint='outcome'
    )
    return ranking_b
