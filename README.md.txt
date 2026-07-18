# Current Progress

Completed so far:
- Step 0: Project foundation and README
- Step 1: Dataset download and folder setup
- Step 2: Raw data loading and dataset structure
- Step 3: Initial sales exploration
- Step 4: Overall sales trend over time
- Step 5: Category, state, and store trends
- Step 6: Event and SNAP effects
- Step 7A: Raw price analysis

Demand-type classification has not been done yet. ADI and CV² will be calculated in Step 11. Until then, all findings are aggregate-level observations.
# MSc Dissertation — Retail Demand Forecasting Using Walmart M5

## Working Title

Retail Demand Forecasting Under Demand Heterogeneity: A Comparative Study Using the Walmart M5 Dataset

## Aim

This dissertation aims to compare different forecasting approaches for item-store level retail demand using the Walmart M5 dataset. The main focus is to understand how forecasting performance changes across different demand pattern types, such as smooth, erratic, intermittent, and lumpy demand.

## Main Research Question

How does demand pattern type affect the forecasting performance of different models for item-store level retail demand in the Walmart M5 dataset?

## Sub-Questions

1. How can item-store sales series be classified into demand types using ADI and CV²?
2. How does a simple seasonal naive baseline perform across different demand types?
3. Does XGBoost improve forecasting accuracy compared with simple baselines?
4. Which feature groups contribute most to forecasting performance?
5. Does an LSTM provide meaningful improvement over XGBoost, or is the added complexity not justified?

## Dataset

The project will use the Walmart M5 Forecasting dataset.

The main files are expected to be:

* sales_train_evaluation.csv or sales_train_validation.csv
* calendar.csv
* sell_prices.csv

The sales file contains daily unit sales at item-store level. The calendar file contains date, event, weekday, month, year, and SNAP-related information. The prices file contains weekly selling prices for each item and store.

## Planned Scope

The full M5 dataset is too large to model fully on a standard laptop, so this project will use a controlled stratified subset of around 120–200 item-store series.

The subset will be selected across four demand types:

* Smooth demand
* Erratic demand
* Intermittent demand
* Lumpy demand

The purpose of this subset is to keep the project computationally manageable while still allowing a fair comparison across different types of retail demand behaviour.

## Planned Models

1. Seasonal naive baseline  
2. XGBoost machine learning model  
3. LSTM targeted feasibility study  

The core comparison is between seasonal naive and XGBoost. The LSTM is included to test whether a simple sequential model adds value over engineered lag features for selected demand types.

## Evaluation Plan

WRMSSE will be used as the primary evaluation metric. MAE will be used as a secondary metric for interpretability.

All final model results will be reported overall and separately by demand type.

## Key Principle

Every result in this dissertation should connect back to demand type.

The project is not only about which model performs best overall. It is about understanding which model works better for which type of retail demand, and why.


Phase A — Foundation and raw data understanding

Step 0: Project foundation and README
Step 1: Download dataset and organise files
Step 2: Load files and understand dataset structure
Step 3: Initial sales exploration
Step 4: Overall sales trend over time
Step 5: Category, state, and store trends
Step 6: Event and SNAP effects
Step 7: Price analysis using sell_prices.csv
Step 8: Deeper zero-sales and sparsity analysis

Phase B — Preparing the modelling dataset

Step 9: Reshape sales from wide to long format
Step 10: Merge sales + calendar + prices
Step 11: Compute ADI and CV² demand taxonomy
Step 12: Select stratified subset across demand types

Phase C — Proper EDA on selected subset

Step 13: Demand-type EDA
Step 14: Final EDA figures and written observations

Phase D — Modelling

Step 15: Seasonal naive baseline
Step 16: Feature engineering
Step 17: XGBoost model
Step 18: Feature ablation and SHAP
Step 19: LSTM feasibility study

Phase E — Final dissertation work

Step 20: Final model comparison, discussion, limitations, and write-up