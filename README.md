# Intensification

This repository was forked from https://github.com/NThakkar-IDM/intensification.git, which is the Python 3.8 code associated with the manuscript [*Routine immunization intensification, vaccination campaigns, and measles transmission in Southern Nigeria*](https://www.medrxiv.org/content/10.1101/2025.02.24.25322796v1), 2025. In that paper, we fit mechanistic transmission models to surveillance and survey data to estimate catch-up vaccination events' effects on measles immunity. We find that the age-targeted 2019 intensification effort was more than twice as effective per dose as the mass campaigns since 2010.

The raw data (the surveys, line list surveillance data, etc.) are not included. Instead, the `_data/` directory includes the processed model inputs, like those visualized in the manuscript's first figure but for every southern state. See [`_data/README.md`](_data/README.md) for a description of each file.

This repository focuses generalizing the transmission model and how to adapt it to other contexts.

## Getting started

You can apply this modeling workflow to other regions and data following the steps below.
The code is pinned to an older scientific Python stack (Python 3.8, numpy 1.18, pandas 1.2; see `environment.yml`), so use the provided environment rather than your base installation.

Either create the conda environment locally:

```
conda env create -f environment.yml
conda activate tsir
```

or open the repository in GitHub Codespaces with the badge at the bottom of this page, which uses the configuration in `.devcontainer/`.

Run all scripts from the repository root, because they read and write relative paths such as `_data/` and `pickle_jar/`.

## Reproduce the paper's figures

For the repository to function correctly, the scripts `GeneratePriors.py` and `GenerateStateSummary.py` have to be run first, in that order. Together they take about 20 minutes on a laptop. These scripts call the others as subprocesses, and they generate serialized output used throughout the repo.

```
python GeneratePriors.py
python GenerateStateSummary.py
```

Then, scripts are set up to create the paper's figures (for Lagos State) by default.

1. `VisualizeInputs.py` makes the paper's Figure 1.
2. `TransmissionModel.py` makes the paper's Figure 2.
3. `OutOfSampleTest.py` makes Figure 3.
4. `SIAImpactPosteriors.py` makes Figure 4.
5. `SIAImpactAnalysis.py` makes Figure 5.
6. `SurvivalPrior.py` makes Appendix 2, Figure 1.

Running these with a different state's name as a command line argument will produce the associated version of the figure for that state. For example, you can try `python TransmissionModel.py oyo`. State names are lowercase, and multi-word names are passed as separate words, as in `python TransmissionModel.py cross river`. The valid names are in the `state` column of `_data/states_and_regions.csv`.

### Where outputs go

- `_plots/` holds the figures as image files (for example `out_of_sample_test.png`).
- `pickle_jar/` holds serialized intermediate results, such as the prior estimates (`coarse_outputs_by_state.pkl`) and the SIA posteriors (`sia_dists_by_state.pkl`).
- Both directories are tracked only for their `.gitignore` files, so their contents are created when you run the scripts. A run has worked if the expected file appears.

If a script fails with a missing-file error for something in `pickle_jar/`, run `GeneratePriors.py` and `GenerateStateSummary.py` first.

## Repository layout

| Path | Contents |
|------|----------|
| `*.py` (repository root) | Scripts that fit models and make figures, as listed above. |
| `methods/` | Functions and classes used by the scripts. See [`methods/README.md`](methods/README.md). |
| `_data/` | Processed model inputs. See [`_data/README.md`](_data/README.md). |
| `_plots/` | Output location for figures (empty until you run the scripts). |
| `pickle_jar/` | Output location for serialized results (empty until you run the scripts). |
| `environment.yml`, `.devcontainer/` | Environment definitions. |

## Use the model with your own data

The transmission model itself is in `methods/neighborhood_sir.py`. Its functions are documented in their docstrings, and `TransmissionModel.py` shows the full call sequence: `prep_model_inputs`, `prep_sia_effects`, `fit_the_neighborhood_model`, then `fit_the_regularized_model`.

To apply the model to another region, you would need to supply, in the same formats as the files in `_data/`:

- A time series of cases, births and routine immunization coverage (see `southern_states_epi_timeseries.csv`),
- A calendar of supplementary immunization activities (SIAs) with dates and doses (`imputed_sia_calendar_by_state.csv`),
- A population by age (`grid3_population_by_state.csv`), and
- Region-level priors on initial susceptibility and the reporting rate, which `GeneratePriors.py` computes for the Southern states.

The scripts in the root directory are written for the Southern Nigerian states and read the state list and file paths directly, so expect to edit them rather than call them unchanged. The workflows that produced these inputs from raw surveys and surveillance data are in `methods/`, for reference.

## Acronyms

- **SIA**: Supplementary immunization activity, such as a mass vaccination campaign.
- **RI**: Routine immunization.
- **MCV1, MCV2**: First and second doses of measles-containing vaccine.
- **DHS, MICS**: The Demographic and Health Surveys and the Multiple Indicator Cluster Surveys, household surveys used to estimate coverage and births.
- **SIR**: Susceptible, infectious, recovered; the compartmental structure of the transmission model.

## License and citation

The code is released under the GNU General Public License v3.0; see [`LICENSE`](LICENSE). If you use it, please cite the manuscript linked above.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/InstituteforDiseaseModeling/intensification)
