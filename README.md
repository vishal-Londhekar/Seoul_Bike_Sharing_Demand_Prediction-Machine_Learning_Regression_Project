# Seoul Bike Sharing Demand Prediction

> **Data Analyst & Business Intelligence Analyst portfolio project**

A machine-learning notebook for predicting bike-sharing demand from weather, calendar, and time-related features.

## Project objective

The project studies demand patterns in Seoul's bike-sharing data and compares regression approaches that can support operational planning.

The workflow covers:

- Data validation and exploratory analysis
- Time and weather feature preparation
- Regression-model training and comparison
- Evaluation of prediction error
- Business interpretation of demand drivers

## Dataset

**SeoulBikeData.csv** contains 8,760 rows and 14 columns according to the project documentation. Review the dataset licence and original source before redistributing it.

## Files

- **Bike_Sharing_Demand_Prediction.ipynb** — analysis and modelling notebook
- **SeoulBikeData.csv** — dataset
- **README.md** — project documentation

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, regression models, and feature engineering.

## How to run

1. Install Python and Jupyter Notebook.
2. Install the packages used by the notebook: NumPy, pandas, Matplotlib, Seaborn, SciPy, and scikit-learn.
3. Keep **SeoulBikeData.csv** in the repository directory.
4. Open **Bike_Sharing_Demand_Prediction.ipynb** and run the cells in order.

## Evaluation

Use the evaluation section in the notebook for the exact train/test split, random seed, baseline, and final metrics. Report the test-set R², MAE, and RMSE together; an R² badge without the evaluation protocol is not sufficient for reproducibility.


## Business problem and decision

### Business problem
Bike-sharing operators must plan capacity around changing demand. Underestimating demand can reduce availability for riders, while overestimating it can waste redistribution and staffing resources.

### Analyst question
How do weather, calendar, season, and time-of-day variables relate to demand, and how accurately can demand be estimated for unseen observations?

### Decision supported
Operations teams can use a validated forecast as one input into bike allocation, staffing, maintenance, and service-planning decisions. The current repository demonstrates the modelling workflow rather than a live planning system.

### Potential success measure
Measure performance on a time-aware holdout using MAE, RMSE, and R², and compare it with a simple baseline before considering operational use.

## Analyst value

> **Portfolio focus:** Operations Analytics · Demand Forecasting · Predictive Decision Support

**Stakeholder lens:** Mobility operators, capacity planners, and operations teams.

**Skills demonstrated:** Data quality checks, time and weather feature engineering, regression modelling, model comparison, error analysis, and operational interpretation.

**Decision support:** Shows how demand estimates can inform staffing, bike availability, and service planning while keeping validation requirements visible.

## Limitations

- This is a notebook-based forecasting prototype, not a live prediction service.
- Performance may change across seasons, weather conditions, and unseen time periods.
- A time-aware validation strategy and a deployed inference pipeline would be required for operational use.

## Author

**Vishal Londhekar** — Data Analyst | Machine Learning Portfolio

[GitHub profile](https://github.com/vishal-Londhekar)