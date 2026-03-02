import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / 'machine_failure_model.pkl'


@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(
            "Model file not found: machine_failure_model.pkl. "
            "Run `Main_Code.py` to train and save the model first."
        )
        return None
    except Exception as error:
        st.error(f"Unable to load model file: {error}")
        return None


model = load_model(MODEL_PATH)
if model is None:
    st.stop()

st.title("Industrial Machine Failure Predictor")
st.write("Enter the sensor readings below to check for potential machine failure.")

# User Inputs
type_val = st.selectbox("Machine Type (L=0, M=1, H=2)", [0, 1, 2])
air_temp = st.number_input("Air Temperature [K]", value=300.0)
proc_temp = st.number_input("Process Temperature [K]", value=310.0)
speed = st.number_input("Rotational Speed [rpm]", value=1500.0)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool Wear [min]", value=0.0)

# Prediction
if st.button("Predict Maintenance Status"):
    # Create input dataframe matching the model's features
    input_data = pd.DataFrame([[type_val, air_temp, proc_temp, speed, torque, tool_wear]],
                              columns=['Type', 'Air temperature [K]', 'Process temperature [K]', 
                                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ PREDICTION: Potential Machine Failure Detected! Maintenance Required.")
    else:
        st.success("✅ PREDICTION: Machine is Operating Normally. No Failure Predicted.")