import streamlit as st
import pandas as pd

# --- Session state check ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- Login Page ---
if not st.session_state.logged_in:
    st.subheader("🔒 Admin Login")

    ad_user = st.text_input("Username")
    ad_password = st.text_input("Password", type='password')

    if st.button("Login"):
        if ad_user == 'TTC' and ad_password == 'ttc@123':
            st.session_state.logged_in = True
            st.success("✅ Login successful! Welcome Manisha 👋")
            st.rerun()  # Refresh page to show dashboard
        else:
            st.error("❌ Invalid username or password")

# --- Dashboard Page (after login) ---
else:
    st.sidebar.write("👤 Logged in as: Manisha")
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update(logged_in=False))
    
    st.title("📊 TTC Delay Dashboard")
    st.write("Welcome! You’re now viewing the main dashboard.")

from ttc_analysis import load_ttc_df, plot_min_delay_boxplot, show_column_names, plot_delay_trend, plot_delay_trend_no_outliers, order_delay_category, avg_delay_of_week, hours_of_day, heat_map, most_common_incidents, delay_distribution, mean_delay, frequent_delays, ttest_weekend_vs_weekday, chisquare_delay_category, anova_avg_delay, ttest_mechanical_issue, top_vs_bottom_routes,prepare_data, linear_regression, random_forest_regressor, gradient_boost_regressor, model_comparison, actual_vs_predicted, plot_rf_feature_importance, plot_xgb_feature_importance
from prediction import prediction_page
ttc_df = load_ttc_df()

# Sidebar navigation menu
page = st.sidebar.selectbox("Choose Page", [
        "Visualizations",
        "Statistical Tests",
        "Predictive Models",
        "Model Comparison",
        "Prediction"
    ])

if page == "Visualizations":
        st.header("Visualizations")
        plot_min_delay_boxplot(ttc_df)
        plot_delay_trend(ttc_df)
        plot_delay_trend_no_outliers(ttc_df)
        order_delay_category(ttc_df)
        avg_delay_of_week(ttc_df)
        hours_of_day(ttc_df)
        heat_map(ttc_df)
        most_common_incidents(ttc_df)
        delay_distribution(ttc_df)
        mean_delay(ttc_df)
        frequent_delays(ttc_df)

elif page == "Statistical Tests":
        st.header("Statistical Tests")
        ttest_weekend_vs_weekday(ttc_df)
        chisquare_delay_category(ttc_df)
        anova_avg_delay(ttc_df)
        ttest_mechanical_issue(ttc_df)
        top_vs_bottom_routes(ttc_df)


elif page == "Predictive Models":
        st.header("Predictive Models")
    
    # Prepare data splits
        X_train, X_test, y_train, y_test = prepare_data(ttc_df)

        try:
            y_test, y_pred_lr = linear_regression(X_train, y_train, X_test, y_test)
        except Exception as e:
            st.error(f"Linear Regression Error: {e}")

        try:
            y_test, y_pred_rg, rf_model = random_forest_regressor(X_train, y_train, X_test, y_test)
        except Exception as e:
            st.error(f"Random Forest Regressor Error: {e}")

        try:
            y_test, y_pred_xgb, xgb_model = gradient_boost_regressor(X_train, y_train, X_test, y_test)
        except Exception as e:
            st.error(f"XGBoost Error: {e}")

        try:
            actual_vs_predicted(y_test, y_pred_lr, y_pred_rg, y_pred_xgb)
            plot_rf_feature_importance(rf_model, X_test)
            plot_xgb_feature_importance(xgb_model)
        except Exception as e:
            st.error(f"Plotting Error: {e}")


elif page == "Model Comparison":
        st.header("Model Comparison")
        model_comparison()

elif page == "Prediction":
        prediction_page(ttc_df) 

