# MSc Dissertation — Retail Demand Forecasting Under Demand Heterogeneity

**Author:** Rishab Kothari (6949800)
**University:** University of Surrey
**Programme:** MSc Data Science
**Module:** COMM070
**Supervisor:** Dr Dany Varghese
**Submission:** September 2026

## Project Title
Retail Demand Forecasting Under Demand Heterogeneity: A Demand-Type-Specific Study Using the Walmart M5 Dataset

## Overview
This repository contains all notebooks and outputs for the MSc dissertation comparing four forecasting models across four demand types using the Walmart M5 dataset.

## Notebooks — Run in This Order
01. Data loading and exploration
02. Demand classification (ADI/CV²)
03. Balanced sampling
04. Feature engineering
05A. Seasonal naive baseline
05B. Croston and SBA baseline
06. XGBoost recursive forecasting
07. Ablation study
08. SHAP analysis
09. LSTM feasibility experiment
10. Classification stability audit

## Data
Raw M5 files not included due to size. Download from:
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Place these in data/raw/:
- sales_train_evaluation.csv
- calendar.csv
- sell_prices.csv

## Key Results
- XGBoost: lowest WRMSSE for smooth (0.6360) and erratic (0.7938)
- SBA: lowest WRMSSE for intermittent (1.0978)
- Croston: lowest WRMSSE for lumpy (1.0674)
- Classification stability: 93.8% agreement across 800 series

## Reproducibility
All random seeds fixed at 42.
Fold dates: outputs/tables/step15/step15_fold_schedule.csv
Saved forecasts: outputs/model_results/

## Repository
https://github.com/rishabrjk/MSc_Dissertation_M5