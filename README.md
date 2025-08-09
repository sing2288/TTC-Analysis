# TTC Delay Analysis Dashboard

## Overview
This project analyzes and predicts delay times in Toronto Transit Commission (TTC) routes using historical transit data. It provides interactive visualizations, statistical tests, and predictive models via a Streamlit dashboard.

---

## Features
- **Visualizations:** Delay trends, distribution, heatmaps, and incident analysis.
- **Statistical Tests:** T-tests, ANOVA, chi-square tests on delay factors.
- **Predictive Models:** Linear Regression, Random Forest, and XGBoost models to predict delay minutes.
- **Model Comparison:** Performance metrics (R², MAE, RMSE) compared across models.
- **Interactive Prediction Page:** Users can input data to predict delay times using trained models.
- **Secure Admin Login:** Dashboard access restricted with username and password.

---

## Dataset
- `TTC_Cleaned.csv` contains cleaned historical TTC delay data.
- Dataset includes features like Route, Min_Gap, Report_Year, Report_Day, and Min_Delay.

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sing2288/TTC-Analysis.git
   cd TTC-Analysis
2. Install dependencies
   pip install -r requirements.txt
3. Run the streamlit app
   streamlit run app.py

## Project Structure 
app.py — Main Streamlit application.

ttc_analysis.py — Contains data loading, visualization, statistical tests, and modeling functions.

prediction.py — Handles the prediction page UI and logic.

TTC_Cleaned.csv — Cleaned dataset.

requirements.txt — Project dependencies.

## Technologies used 

Python 3

Pandas, NumPy for data processing

Scikit-learn for modeling

Streamlit for interactive dashboard

Matplotlib and Seaborn for visualizations

Git and GitHub for version control

## License 
This project is for eductaional purposes only.


