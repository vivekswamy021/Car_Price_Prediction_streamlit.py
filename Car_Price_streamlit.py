# streamlit app
import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Config ---
st.set_page_config(page_title="Vehicle Price Predictor", page_icon="🚗")

# --- Load Model and Encoders ---
@st.cache_resource
def load_assets():
    # Adding checks to ensure files exist before loading
    model_path = "linear_regression_model.joblib"
    encoder_path = "label_encoders.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        st.error("Model or Encoder files missing! Please ensure .joblib files are in the directory.")
        return None, None
        
    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    return model, encoders

model, encoders = load_assets()

# --- App Title and Description ---
st.title("🚗 Vehicle Price Prediction")
st.markdown("Enter the vehicle details in the sidebar to get an estimated selling price.")

# --- User Input in Sidebar ---
if encoders:
    st.sidebar.header("Vehicle Features")

    def user_input_features():
        # Using a dictionary to collect inputs
        data = {}
        data["year"] = st.sidebar.slider("Year", 1990, 2026, 2018)
        
        # Categorical inputs from encoders
        categorical_cols = ["make", "model", "trim", "body", "transmission", "state", "color", "interior", "seller"]
        
        for col in categorical_cols:
            if col in encoders:
                data[col] = st.sidebar.selectbox(f"{col.title()}", encoders[col].classes_)
        
        # Numerical inputs
        data["condition"] = st.sidebar.slider("Condition (1-5)", 1.0, 5.0, 3.5, 0.1)
        data["odometer"] = st.sidebar.number_input("Odometer (miles)", min_value=0, max_value=500000, value=50000)
        data["mmr"] = st.sidebar.number_input("Manheim Market Report (MMR)", min_value=0, value=20000)

        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # --- Display Inputs ---
    st.subheader("Selected Vehicle Specs")
    st.dataframe(input_df)

    # --- Prediction Logic ---
    if st.button("Predict Price"):
        try:
            # Encode User Inputs
            encoded_df = input_df.copy()
            for col in encoders:
                if col in encoded_df.columns:
                    encoded_df[col] = encoders[col].transform(encoded_df[col])
            
            # Predict
            prediction = model.predict(encoded_df)
            
            # Display Result
            st.markdown("---")
            st.subheader("Estimated Market Value")
            price = max(0, prediction[0]) # Ensure price isn't negative
            st.success(f"### **${price:,.2f}**")
            
            # Contextual info
            st.info(f"The predicted price is based on a condition score of {input_df['condition'].iloc[0]}/5.")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
else:
    st.warning("Please upload your `linear_regression_model.joblib` and `label_encoders.joblib` files to the repository.")
