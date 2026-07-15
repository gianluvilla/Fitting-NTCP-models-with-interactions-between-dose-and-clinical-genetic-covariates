## Overview

This repository contains the code used to implement the methodology presented in the paper **“Cellular versus genetic radiosensitivity: distinct predictive roles across dose ranges. Analysis of late post-radiotherapy overreaction in prostate cancer patients”**.

It includes the functions required to fit and evaluate the proposed NTCP model and rank SNPs according to their relevance to the outcome. The model incorporates dose, genetic risk, radiosensitivity, and their interactions.

The methodology is illustrated using synthetic datasets containing 100 and 1000 patients. The repository also includes the code used to generate these datasets and their corresponding outcome variables.

## Repository structure

```text
NTCP_with_interactions/
│
├── data/
│   ├── synth_data_100pts.xlsx                     # Synthetic data used in the 100-patient examples
│   └── synth_data_1000pts.xlsx                    # Synthetic data used in the 1000-patient examples
│
├── notebooks/
│   ├── ntcp_model_fitting/
│   │   ├── ntcp_model_on_synthetic_data_100pts.ipynb   # Fit and evaluate the NTCP model on 100 patients
│   │   └── ntcp_model_on_synthetic_data_1000pts.ipynb  # Fit and evaluate the NTCP model on 1000 patients
│   │
│   ├── SNPs_ranking/
│   │   ├── SNPs_ranking_100pts.ipynb              # Apply the SNP-ranking procedure to the smaller dataset
│   │   ├── SNPs_ranking_1000pts.ipynb             # Apply the SNP-ranking procedure to the larger dataset
│   │   └── SNPs_ranking_explicit_example.ipynb    # Test the ranking with predefined SNP–outcome associations
│   │
│   └── synthetic_data_generation/
│       ├── synth_data_generation_100pts.ipynb      # Generate the synthetic variables and outcomes for 100 patients
│       └── synth_data_generation_1000pts.ipynb     # Generate the synthetic variables and outcomes for 1000 patients
│
├── src/
│   ├── ntcp_fit/
│   │   ├── __init__.py
│   │   ├── bootstrap.py
│   │   ├── config.py
│   │   ├── goodness_of_fit.py
│   │   ├── optimization.py
│   │   └── utilities.py
│   │
│   ├── SNPs_selection/
│   │   ├── __init__.py
│   │   └── snps_selection.py
│   │
│   └── synth_data/
│       ├── __init__.py
│       └── outcome_generation.py
│
├── .gitignore
├── LICENSE
└── README.md
```


- `src/ntcp_fit/`: package containing the functions required to configure, fit, and evaluate the NTCP model. It includes parameter optimization, bootstrap analysis, goodness-of-fit tests, and supporting utilities.

- `notebooks/ntcp_model_fitting/`: practical examples showing the complete model-fitting workflow on synthetic cohorts of 100 and 1000 patients. The notebooks include model configuration, parameter estimation, and performance evaluation.

- `src/SNPs_selection/`: package containing the functions used to assess the relevance of the available SNPs and rank them according to their contribution to the model.

- `notebooks/SNPs_ranking/SNPs_ranking_100pts.ipynb`: applies the SNP-ranking procedure to the synthetic dataset containing 100 patients.

- `notebooks/SNPs_ranking/SNPs_ranking_1000pts.ipynb`: applies the same SNP-ranking procedure to the larger synthetic dataset containing 1000 patients.

- `notebooks/SNPs_ranking/SNPs_ranking_explicit_example.ipynb`: illustrative example in which the outcome is generated to depend strongly on `SNP_1` and `SNP_2`, but not on `SNP_3` and `SNP_4`. The resulting ranking is expected to show a clear elbow-shaped separation between associated and non-associated SNPs.

- `src/synth_data/`: package containing the functions used to generate synthetic outcomes based on dose, genetic risk, radiosensitivity, and their interactions.

- `notebooks/synthetic_data_generation/`: notebooks describing how the synthetic patient characteristics and outcome variables are generated for cohorts of 100 and 1000 patients.

- `data/synth_data_100pts.xlsx`: synthetic dataset generated for the examples based on a cohort of 100 patients.

- `data/synth_data_1000pts.xlsx`: synthetic dataset generated for the examples based on a cohort of 1000 patients.


## Dependencies

The project requires Python and the following external packages:

- `numpy`: numerical operations and synthetic data generation.
- `pandas`: dataset creation and manipulation.
- `scipy`: parameter optimization and statistical calculations.
- `matplotlib`: plots of results and generated data.
- `statsmodels`: statistical tests and model diagnostics.
- `scikit-learn`: data scaling and performance metrics.
- `joblib`: parallel implementation.,
- `openpyxl`: reading and writing Excel files.


<!-- ## Citation

When using this code, please cite the associated article:

```text
<<COMPLETE_ARTICLE_CITATION>>
```

BibTeX:

```bibtex
@article{<<BIBTEX_KEY>>,
    author  = {<<AUTHORS>>},
    title   = {<<ARTICLE_TITLE>>},
    journal = {<<JOURNAL>>},
    year    = {<<YEAR>>},
    volume  = {<<VOLUME>>},
    number  = {<<ISSUE>>},
    pages   = {<<PAGES_OR_ARTICLE_NUMBER>>},
    doi     = {<<ARTICLE_DOI>>}
}
``` -->

## License
This project is licensed under the MIT [License](LICENSE).

## Contact

Name: Gianluca Villa<br>
Institution: Fondazione IRCCS Istituto Nazionale dei Tumori (Milan, Italy)<br>
E-mail: Gianluca.Villa@istitutotumori.mi.it