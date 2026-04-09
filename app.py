import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model("cnn_lstm_model.h5", compile=False)

st.title("🔧 Predictive Maintenance - RUL Dashboard")

timesteps = 30
features = 22

tab1, tab2, tab3 = st.tabs(["Inputs", "Prediction", "Visualization"])

with tab1:
    st.subheader("Sensor Inputs")
    cols = st.columns(3)
    sensor_values = []
    for i in range(features):
        with cols[i % 3]:
            val = st.slider(f"Sensor {i+1}", 0.0, 1.0, 0.5)
            sensor_values.append(val)
    input_array = np.array(sensor_values * timesteps).reshape(1, timesteps, features)

with tab2:
    st.subheader("RUL Prediction")
    if st.button("Predict RUL"):
        prediction = model.predict(input_array)
        rul = prediction[0][0]
        st.metric(label="Predicted RUL (cycles)", value=f"{rul:.2f}")
        st.write(f"Expected operation: **{rul:.0f} more cycles** before maintenance.")
        st.progress(min(int(rul), 500))

with tab3:
    st.subheader("Sensor Trend")
    fig, ax = plt.subplots()
    ax.plot(input_array.flatten()[:100], color="blue")
    ax.set_title("Sensor Trend (first 100 values)")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Sensor value")
    ax.grid(True)
    st.pyplot(fig)
