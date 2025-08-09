
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import re
import sqlalchemy
import pyodbc
import streamlit as st
import calendar



def load_ttc_df():
    ttc_df = pd.read_csv("TTC_Cleaned.csv")  # <- not TTC.csv
    return ttc_df



def plot_min_delay_boxplot(ttc_df):
    st.subheader("📦 Box Plot of Minimum Delays")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw a box plot of the 'Min_Delay' column
    ax.boxplot(ttc_df['Min_Delay'])
    
    # Set custom Y-axis range
    ax.set_ylim(0, ttc_df['Min_Delay'].quantile(0.95))  # Adjust the upper limit as needed
    
    # Set plot title and labels
    ax.set_title('Box Plot of Min_Delay')
    ax.set_ylabel('Minutes')
    
    # Show the plot in Streamlit
    st.pyplot(fig)





def show_column_names(ttc_df):
    st.subheader("🧾 Column Names in the Dataset")
    st.write(list(ttc_df.columns))

sns.set_style("whitegrid")

# %%
# Plot Trend Analysis: Delays over time
def plot_delay_trend(ttc_df):
    st.subheader("📈 Trend of Delays Over Time")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='Report_Date', y='Min_Delay', data=ttc_df, errorbar=None, ax=ax)

    ax.set_title('Trend of Delays Over Time')
    ax.set_xlabel('Report Date')
    ax.set_ylabel('Min Delay')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=45)

    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

    fig.tight_layout()
    st.pyplot(fig)

# Filter out delays greater than 200 minutes
def plot_delay_trend_no_outliers():
    st.subheader("📉 Delay Trend (Excluding Outliers)")

    filtered_ttc_df = ttc_df[ttc_df['Min_Delay'] <= 200]

# Plot Trend Analysis: Delays over time

def plot_delay_trend_no_outliers(ttc_df):
    st.subheader("📉 Trend of Delays Over Time (Excluding Outliers)")

    # Filter out extreme delays
    filtered_ttc_df = ttc_df[ttc_df['Min_Delay'] <= 200]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='Report_Date', y='Min_Delay', data=filtered_ttc_df, errorbar=None, ax=ax)

    ax.set_title('Trend of Delays Over Time (Excluding Outliers)')
    ax.set_xlabel('Report Date')
    ax.set_ylabel('Min Delay')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=45)

    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

    fig.tight_layout()
    st.pyplot(fig)


# Define the order of delay categories
def order_delay_category(ttc_df):
    st.subheader("📊 Distribution of Delay Categories")

    delay_category_order = ['Short', 'Medium', 'Long']

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Delay_Category', data=ttc_df, order=delay_category_order, ax=ax)

    ax.set_title('Distribution of Delay Categories')
    ax.set_xlabel('Delay Category')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    # Add count labels
    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2.,
            p.get_height(),
            '{:,.0f}'.format(p.get_height()),
            fontsize=12,
            color='black',
            ha='center',
            va='bottom'
        )

    st.pyplot(fig)

# ## Which day of the week is the busiest? 
# Average Delay by Day of the Week

def avg_delay_of_week(ttc_df):
    st.subheader("📅 Average Delay by Day of the Week")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x='Day_of_week',
        y='Min_Delay',
        data=ttc_df,
        ci=None,
        order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        ax=ax
    )

    ax.set_title("Average Delay by Day of the Week")
    ax.set_xlabel("Days of Week")
    ax.set_ylabel("Min Delay")
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=45)

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2f}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 10),
            textcoords='offset points'
        )

    st.pyplot(fig)

# # Which hours of the day are the busiest ?

def hours_of_day(ttc_df):
    st.subheader("⏰ Frequency of Delays by Hour of Day")

    # Create a copy to avoid modifying original
    temp_df = ttc_df.copy()
    temp_df['Hour'] = pd.to_datetime(temp_df['Time'], errors='coerce').dt.hour

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(temp_df['Hour'].dropna(), bins=24, kde=False, color='skyblue', ax=ax)

    ax.set_title('Frequency of Delays by Hour of Day')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Delays')
    ax.set_xticks(range(0, 24))

    fig.tight_layout()
    st.pyplot(fig)

# # Delay Heat map by Month and Year 
def heat_map(ttc_df):
    st.subheader("🌡️ Delay Heatmap by Month and Year")

    # Create pivot table of average delay
    pivot_table = ttc_df.pivot_table(
        index='Report_Month',
        columns='Report_Year',
        values='Min_Delay',
        aggfunc='mean'
    )

    # List of month abbreviations for y-axis labels
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_table,
        cmap='YlGnBu',
        annot=True,
        fmt=".1f",
        cbar_kws={'label': 'Average Min Delay'},
        ax=ax
    )

    ax.set_title('Delay Heatmap by Month and Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Month')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticks(range(12))
    ax.set_yticklabels(month_names, rotation=0)

    fig.tight_layout()
    st.pyplot(fig)


# ## Which is the most common incidents all?

def most_common_incidents(ttc_df):
    st.subheader("🚨 Distribution of Delay Incidents")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        x='Incident', 
        data=ttc_df, 
        order=ttc_df['Incident'].value_counts().index, 
        ax=ax
    )

    ax.set_title('Distribution of Delay Incidents')
    ax.set_xlabel('Incident Type')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)

# # What is the delay distribution of top 10 routes?
def delay_distribution(ttc_df):
    st.subheader("🚦 Delay Distribution by Top 10 Routes")

    top_routes = ttc_df['Route'].value_counts().head(10).index

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        x='Route',
        y='Min_Delay',
        data=ttc_df[ttc_df['Route'].isin(top_routes)],
        ax=ax
    )

    ax.set_title('Delay Distribution by Top 10 Routes')
    ax.set_xlabel('Route')
    ax.set_ylabel('Min Delay')
    ax.tick_params(axis='x', rotation=45)

    fig.tight_layout()
    st.pyplot(fig)

# # What is the mean delay for top 10 routes?

def mean_delay(ttc_df):
    st.subheader("🚦 Top 10 Routes with Highest Average Delays")

    # Calculate average delay by route
    route_avg_delay = ttc_df.groupby('Route')['Min_Delay'].mean()

    # Get top 10 routes sorted descending
    top_routes_desc = route_avg_delay.sort_values(ascending=False).head(10).index

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        x='Route',
        y='Min_Delay',
        data=ttc_df[ttc_df['Route'].isin(top_routes_desc)],
        ci=None,
        order=top_routes_desc,
        ax=ax
    )

    ax.set_title('Top 10 Routes with Highest Average Delays (Descending order)')
    ax.set_xlabel('Route')
    ax.set_ylabel('Average Min Delay')
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)

# ## Top 10 locations with Most Frequent Delays

def frequent_delays(ttc_df):
    st.subheader("📍 Top 10 Locations with Most Frequent Delays")

    top_locations = ttc_df['Location'].value_counts().index[:10]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        x='Location',
        data=ttc_df,
        order=top_locations,
        ax=ax
    )

    ax.set_title('Top 10 Locations with Most Frequent Delays')
    ax.set_xlabel('Location')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)


# %%
# T-test Are weekend delays significantly different from weekday delays?

from scipy.stats import ttest_ind

def ttest_weekend_vs_weekday(ttc_df):
    st.subheader("📊 T-Test: Weekend vs Weekday Delays")

    # Create 'is_weekend' column
    ttc_df['is_weekend'] = ttc_df['Day_of_week'].isin(['Saturday', 'Sunday'])

    weekend = ttc_df[ttc_df['is_weekend']]['Min_Delay']
    weekday = ttc_df[~ttc_df['is_weekend']]['Min_Delay']

    # Perform t-test
    t_stat, p_val = ttest_ind(weekday, weekend, nan_policy='omit')

    # Display results
    st.write(f"**T-Statistic:** {t_stat:.3f}")
    st.write(f"**P-Value:** {p_val:.4f}")

    # Interpretation
    if p_val < 0.05:
        st.success("✅ There is a statistically significant difference in delays between weekdays and weekends.")
    else:
        st.info("ℹ️ There is **no** statistically significant difference in delays between weekdays and weekends.")


# Chi-squaretest Is delay category dependent on day type?
from scipy.stats import chi2_contingency

def chisquare_delay_category(ttc_df):
    st.subheader("📊 Chi-Square Test: Delay Category vs Weekend")

    # Ensure 'is_weekend' column exists
    if 'is_weekend' not in ttc_df.columns:
        ttc_df['is_weekend'] = ttc_df['Day_of_week'].isin(['Saturday', 'Sunday'])

    # Create contingency table
    delay_day_ct = pd.crosstab(ttc_df['Delay_Category'], ttc_df['is_weekend'])

    # Perform chi-square test
    chi2, p, dof, expected = chi2_contingency(delay_day_ct)

    # Show results
    st.write("**Contingency Table:**")
    st.dataframe(delay_day_ct)

    st.write(f"**Chi² Statistic:** {chi2:.2f}")
    st.write(f"**Degrees of Freedom:** {dof}")
    st.write(f"**P-value:** {p:.4f}")

    # Interpretation
    if p < 0.05:
        st.success("✅ Statistically significant association between delay category and weekend/weekday.")
    else:
        st.info("ℹ️ No statistically significant association between delay category and weekend/weekday.")


#Anova Does average delay differ across different months?
from scipy.stats import f_oneway

def anova_avg_delay(ttc_df):
    st.subheader("📅 ANOVA: Delay Differences Across Months")

    # Group Min_Delay by month
    groups = [group['Min_Delay'].dropna().values for name, group in ttc_df.groupby('Report_Month')]

    # Run ANOVA test
    f_stat, p_val = f_oneway(*groups)

    # Show results
    st.write(f"**F-statistic:** {f_stat:.2f}")
    st.write(f"**P-value:** {p_val:.4f}")

    # Interpretation
    if p_val < 0.05:
        st.success("✅ Statistically significant difference in average delays across months.")
    else:
        st.info("ℹ️ No statistically significant difference in delays across months.")




# %%
# T-Test: Do mechanical issues result in longer delays than general delays?

def ttest_mechanical_issue(ttc_df):
    st.subheader("🔧 T-Test: Mechanical vs General Delays")

    # Subset the data
    mechanical_delays = ttc_df[ttc_df['Incident'] == 'Mechanical']['Min_Delay'].dropna()
    general_delays = ttc_df[ttc_df['Incident'] == 'Delay']['Min_Delay'].dropna()

    # Perform Welch’s t-test
    t_stat, p_val = ttest_ind(mechanical_delays, general_delays, equal_var=False)

    # Display results
    st.write(f"**T-statistic:** {t_stat:.2f}")
    st.write(f"**P-value:** {p_val:.4f}")

    # Interpretation
    if p_val < 0.05:
        st.success("✅ Statistically significant difference between mechanical and general delay durations.")
    else:
        st.info("ℹ️ No statistically significant difference between mechanical and general delay durations.")



# %%
#T-test: Top 10 Routes vs Bottom 10 Routes Delay Duration

def top_vs_bottom_routes(ttc_df):
    st.subheader("🛣️ T-Test: Top 10 vs Bottom 10 Routes (by Frequency)")

    # Identify top and bottom routes
    top_routes = ttc_df['Route'].value_counts().head(10).index
    bottom_routes = ttc_df['Route'].value_counts().tail(10).index

    # Subset delays and drop missing values
    top_delays = ttc_df[ttc_df['Route'].isin(top_routes)]['Min_Delay'].dropna()
    bottom_delays = ttc_df[ttc_df['Route'].isin(bottom_routes)]['Min_Delay'].dropna()

    # Perform Welch's t-test
    t_stat, p_val = ttest_ind(top_delays, bottom_delays, equal_var=False)

    # Display results
    st.write(f"**T-statistic:** {t_stat:.2f}")
    st.write(f"**P-value:** {p_val:.4f}")

    # Interpretation
    if p_val < 0.05:
        st.success("✅ Statistically significant difference in delay times between top 10 and bottom 10 routes.")
    else:
        st.info("ℹ️ No statistically significant difference in delay times between top 10 and bottom 10 routes.")

# ## Predictive Modeling 


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

# Select features and target
def prepare_data(ttc_df):
    features = ['Route', 'Min_Gap', 'Report_Year', 'Report_Day']
    X = pd.get_dummies(ttc_df[features], drop_first=True)
    y = ttc_df['Min_Delay']
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.2, random_state=42)

# %%
#Linear Regression( for delay minutes)


def linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred_lr = model.predict(X_test)

    # Evaluation
    r2 = r2_score(y_test, y_pred_lr)
    mae = mean_absolute_error(y_test, y_pred_lr)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    # Display in Streamlit
    st.subheader("📊 Linear Regression Evaluation")
    st.metric("R² Score", f"{r2:.3f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")

    return y_test, y_pred_lr


# %%

# %%
# Random Foret regressor 

def random_forest_regressor(X_train, y_train, X_test, y_test):
    # Train model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict
    y_pred_rg = rf_model.predict(X_test)

    # Evaluation
    r2 = r2_score(y_test, y_pred_rg)
    mae = mean_absolute_error(y_test, y_pred_rg)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_rg))

    # Display results in Streamlit
    st.subheader("🌲 Random Forest Regressor Evaluation")
    st.metric("R² Score", f"{r2:.3f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")

    return y_test, y_pred_rg, rf_model



# %%
# Gradient Boost Regressor
import streamlit as st
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def gradient_boost_regressor(X_train, y_train, X_test, y_test):
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42
    }

    # Train model
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    # Predict
    y_pred_xgb = xgb_model.predict(dtest)

    # Evaluation
    r2 = r2_score(y_test, y_pred_xgb)
    mae = mean_absolute_error(y_test, y_pred_xgb)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

    # Display in Streamlit
    st.subheader("🚀 Gradient Boosting (XGBoost) Evaluation")
    st.metric("R² Score", f"{r2:.3f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")

    return y_test, y_pred_xgb, xgb_model


# Model Performance Comparison Bar Chart

def model_comparison():
    # Replace with your actual metrics
    model_metrics = {
        'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
        'R2 Score': [0.9291, -10.4395, 0.8790],
        'MAE': [6.11, 3.60, 3.21],
        'RMSE': [18.63, 236.68, 24.35]
    }

    df_metrics = pd.DataFrame(model_metrics)

    # R² Score Plot
    st.subheader("📈 R² Score Comparison")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Model', y='R2 Score', data=df_metrics, palette='Blues_d', ax=ax1)
    ax1.set_ylim(0, 1)
    st.pyplot(fig1)

    # MAE Plot
    st.subheader("📊 MAE Comparison")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Model', y='MAE', data=df_metrics, palette='Greens_d', ax=ax2)
    st.pyplot(fig2)

    # RMSE Plot
    st.subheader("📉 RMSE Comparison")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Model', y='RMSE', data=df_metrics, palette='Reds_d', ax=ax3)
    st.pyplot(fig3)

# Actual Vs Predicted plot 
def actual_vs_predicted(y_test, y_pred_lr, y_pred_rg, y_pred_xgb):
    st.subheader("🎯 Actual vs Predicted Delay Comparison")

    # Set style
    sns.set(style="whitegrid")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Linear Regression
    sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.6, color='blue', ax=axes[0])
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    axes[0].set_title('Linear Regression')
    axes[0].set_xlabel('Actual Delay')
    axes[0].set_ylabel('Predicted Delay')

    # Random Forest
    sns.scatterplot(x=y_test, y=y_pred_rg, alpha=0.6, color='green', ax=axes[1])
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    axes[1].set_title('Random Forest')
    axes[1].set_xlabel('Actual Delay')
    axes[1].set_ylabel('Predicted Delay')

    # XGBoost
    sns.scatterplot(x=y_test, y=y_pred_xgb, alpha=0.6, color='orange', ax=axes[2])
    axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    axes[2].set_title('XGBoost')
    axes[2].set_xlabel('Actual Delay')
    axes[2].set_ylabel('Predicted Delay')

    plt.tight_layout()
    st.pyplot(fig)


# Random forest feature importance 

def plot_rf_feature_importance(rf_model, X):
    st.subheader("🌲 Top Feature Importances (Random Forest)")

    # Get importances
    importances = rf_model.feature_importances_
    features = X.columns

    # Create DataFrame
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis', ax=ax)
    ax.set_title('Top Feature Importances')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()

    st.pyplot(fig)

def plot_xgb_feature_importance(xgb_model, max_features=15):
    st.subheader("🚀 XGBoost Feature Importances")

    # Capture the plot into a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(
        xgb_model,
        max_num_features=max_features,
        height=0.5,
        ax=ax
    )
    ax.set_title('XGBoost Feature Importance')
    plt.tight_layout()
    st.pyplot(fig)



