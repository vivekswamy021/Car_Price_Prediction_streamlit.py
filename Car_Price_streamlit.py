import streamlit as st
import pandas as pd
import joblib

# --- Load Model and Encoders ---
# We use st.cache_resource so the model only loads once, saving memory.
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("linear_regression_model.joblib")
        encoders = joblib.load("label_encoders.joblib")
        return model, encoders
    except FileNotFoundError:
        st.error("Model files not found. Please ensure .joblib files are in the repository.")
        return None, None

model, encoders = load_assets()

# --- App UI ---
st.set_page_config(page_title="Vehicle Price Predictor", page_icon="🚗")
st.title("🚗 Vehicle Price Prediction")
st.markdown("Adjust the features in the sidebar to estimate the selling price.")

if model and encoders:
    # --- Sidebar Inputs ---
    st.sidebar.header("Vehicle Features")

    def get_user_input():
        year = st.sidebar.slider("Year", 1990, 2026, 2018)
        # Using .classes_ from your saved LabelEncoders
        make = st.sidebar.selectbox("Make", encoders["make"].classes_)
        model_input = st.sidebar.selectbox("Model", encoders["model"].classes_)
        trim = st.sidebar.selectbox("Trim", encoders["trim"].classes_)
        body = st.sidebar.selectbox("Body Type", encoders["body"].classes_)
        transmission = st.sidebar.selectbox("Transmission", encoders["transmission"].classes_)
        state = st.sidebar.selectbox("State", encoders["state"].classes_)
        condition = st.sidebar.slider("Condition (1-5)", 1.0, 5.0, 3.5, 0.1)
        odometer = st.sidebar.number_input("Odometer (miles)", 0, 500000, 50000)
        color = st.sidebar.selectbox("Color", encoders["color"].classes_)
        interior = st.sidebar.selectbox("Interior Color", encoders["interior"].classes_)
        seller = st.sidebar.selectbox("Seller", encoders["seller"].classes_)
        mmr = st.sidebar.number_input("Manheim Market Report (MMR)", 0, 100000, 20000)

        data = {
            "year": year, "make": make, "model": model_input, "trim": trim,
            "body": body, "transmission": transmission, "state": state,
            "condition": condition, "odometer": odometer, "color": color,
            "interior": interior, "seller": seller, "mmr": mmr
        }
        return pd.DataFrame(data, index=[0])

    input_df = get_user_input()

    # --- Display Inputs ---
    st.subheader("Selected Vehicle Specs")
    st.dataframe(input_df)

    # --- Encoding & Prediction ---
    if st.button("Predict Price"):
        # Create a copy for encoding to keep the original for display
        encoded_df = input_df.copy()
        
        for col, encoder in encoders.items():
            if col in encoded_df.columns:
                val = encoded_df[col].iloc[0]
                try:
                    encoded_df[col] = encoder.transform([val])
                except ValueError:
                    # Fallback for unseen labels
                    encoded_df[col] = -1 

        # Ensure column order matches training (important for Linear Regression)
        # Assuming the model expects columns in a specific order:
        prediction = model.predict(encoded_df)
        
        st.divider()
        st.subheader("Estimated Price")
        # Ensure the prediction isn't negative (common with simple LinReg)
        final_price = max(0, prediction[0])
        st.success(f"The estimated selling price is: **${final_price:,.2f}**")
