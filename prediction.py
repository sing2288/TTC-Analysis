import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

def prediction_page(ttc_df):
    st.header("Prediction Page")

    route_options = sorted(ttc_df['Route'].unique())
    route = st.selectbox("Select Route", route_options)
    min_gap = st.number_input("Min Gap", min_value=0, value=int(ttc_df['Min_Gap'].mean()))
    report_year = st.number_input("Report Year", min_value=int(ttc_df['Report_Year'].min()), max_value=int(ttc_df['Report_Year'].max()), value=int(ttc_df['Report_Year'].mean()))
    report_day = st.number_input("Report Day", min_value=1, max_value=31, value=15)

    if st.button("Predict Delay"):
        input_df = pd.DataFrame({
            'Route': [route],
            'Min_Gap': [min_gap],
            'Report_Year': [report_year],
            'Report_Day': [report_day]
        })

        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Make sure columns match training data columns
        X_full = pd.get_dummies(ttc_df[['Route', 'Min_Gap', 'Report_Year', 'Report_Day']], drop_first=True)
        for col in X_full.columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[X_full.columns]

        model = LinearRegression()
        model.fit(X_full, ttc_df['Min_Delay'])

        prediction = model.predict(input_encoded)[0]
        st.success(f"Predicted Minimum Delay (minutes): {round(prediction, 2)}")
