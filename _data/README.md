# Model inputs

These are the processed inputs used by the scripts in the repository root. They were produced from raw survey, surveillance and population data that are not included here, using the workflows in `../methods/`. Each file has an unnamed first column holding the row index, which you can ignore or drop with `index_col=0`.

Columns named `var` or ending in `_var` are variances of the corresponding estimate, not standard deviations. States are lowercase, and the valid names are listed in `states_and_regions.csv`.

| File | Rows | Description | Columns |
|------|------|-------------|---------|
| `states_and_regions.csv` | One per state | Maps each Nigerian state (plus Abuja) to a geopolitical region. | `state`, `region` |
| `southern_states_epi_timeseries.csv` | One per state per day, from 2008 | Daily model inputs by state. Used by `TransmissionModel.py`, `OutOfSampleTest.py` and others. | `state`, `time`, `births`, `births_var` (births per day and variance), `mcv1`, `mcv1_var`, `mcv2`, `mcv2_var` (routine coverage and variance), `cases`, `cases_var` (estimated measles cases and variance), `rejected` (suspected cases ruled out as measles) |
| `imputed_sia_calendar_by_state.csv` | One per SIA per state | Calendar of supplementary immunization activities, with missing details imputed. Read by `SIACalendar.py` and `prep_sia_effects(...)`. | `state`, `start_date`, `end_date`, `age_group` (for example `9M-15Y`), `doses` |
| `grid3_population_by_state.csv` | One per state per age bin | Population by age and sex, from GRID3. Read by `get_age_pyramid(...)`. | `state`, `age_bin` (for example `0 to 1`), `female`, `male`, `total` |
| `monthly_births_by_state.csv` | One per state per month | Estimated births per month. | `state`, `time`, `avg`, `var` |
| `birth_seasonality_profiles.csv` | One per state per birth year and month | Seasonal share of births within each year. | `state`, `birth_year`, `birth_month`, `avg`, `var` |
| `survey_mcv1_summary_stats.csv` | One per state, year, age group and survey type | Survey-based MCV1 coverage estimates. | `state`, `year`, `age_group` (for example `12_23m`), `svy_type` (for example `DHS`), `mcv_est`, `mcv_se` (estimate and standard error) |
| `southern_age_at_infection.csv` | One per region, birth year and age | Estimated distribution of age at infection by birth cohort. | `region`, `birth_year`, `age`, `avg`, `var` |
| `binned_age_distributions.csv` | One per state | Age distribution in yearly bins, as fractions that sum to 1 across the columns. | `state`, then one column per age in years (`0.0` to `14.0`) |

## Sources

From the repository, the population data come from GRID3, and the survey data from the DHS and MICS programs. The surveillance and SIA information is described in the manuscript's methods.
