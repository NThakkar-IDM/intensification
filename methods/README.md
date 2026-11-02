# Methods

This directory contains some functions and classes used throughout the scripts in `intensification`.

From the perspective of the [manuscript](https://www.medrxiv.org/content/10.1101/2025.02.24.25322796v1), the only part of this library used directly is `neighborhood_sir.py`. The other scripts illustrate data processing workflows used to make the processed outputs in `../_data/`, but the raw data needed to run them is *not* included. They're here in this repository for reference purposes only, in case they help others apply these models and ideas to their own work.

## Workflows

There are three main workflows included:

1. `epi_curves`, which covers the logistic regression applied at individual level to estimate probabilities that untested, clinically compatible cases are actually measles. This is the analysis described in the manuscript's appendix 1.
2. `age_at_inf`, which covers the smoothing process used to estimate the age distribution of infections by birth cohort, referenced in the manuscript's second appendix.
3. `demography`, which covers the multiple-regression with post-stratification methods applied to the Demographic and Health Surveys (DHS) and Multiple Indicator Cluster Surveys (MICS) to estimate birthrates and routine coverage. Outputs are like those visualized in the manuscript's Figure 1 for Lagos State.

## Files

The Python files use a mix of CamelCase and snake_case names. This table is a guide to what each one does. Details are in the module docstrings.

| Path | Purpose |
|------|---------|
| `neighborhood_sir.py` | The transmission model: `NeighborhoodPosterior`, `HoodRegularizedModel`, input preparation and fitting functions. Documented in its docstrings. |
| `epi_curves/EpiCurves.py` | Builds epidemic curves of cases by state. |
| `epi_curves/ConfRates.py` | Estimates the rate at which clinically compatible cases are confirmed as measles. |
| `epi_curves/logistic.py` | Logistic regression helpers for the confirmation model. |
| `age_at_inf/StratifiedDistribution.py` | Estimates the age-at-infection distribution by birth cohort. |
| `age_at_inf/masked_buckets.py` | Helpers for age buckets with missing or masked values. |
| `demography/BirthSeasonality.py`, `MonthlyBirths.py`, `YearlyBirths.py` | Birth counts and their seasonality. |
| `demography/AgeAtKthKid.py`, `MomDistribution.py`, `ZeroInflatedNumKids.py` | Number of children and mother's age at each birth. |
| `demography/MCVOneAge.py`, `MCVOneProbability.py`, `MCVOneCoverageEstimates.py` | Age at, probability of, and overall coverage of the first measles vaccine dose. |
| `demography/survey/` | Shared survey regression tools (logistic, negative binomial, zero-inflated negative binomial and ridge regressions, plus age bucketing and mapping).
