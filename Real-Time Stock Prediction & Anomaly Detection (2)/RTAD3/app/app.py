

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model

# Fixed model path (matches Docker volume mount)
MODEL_PATH = "/app/model/lstm_model.h5"

def load_keras_model():
    """Load the pre-trained Keras model with caching"""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def main():
    st.title("Stock Prediction Dashboard")
    st.markdown("Using pre-trained model at {}".format(MODEL_PATH))

    # Load model once at startup
    model = load_keras_model()
    
    if model is None:
        st.stop()

    # Display model information
    with st.expander("Model Details"):
        st.subheader("Model Architecture")
        st.text(model.summary())
        st.write("Input shape:", model.input_shape)
        st.write("Output shape:", model.output_shape)

    # Prediction interface
    st.header("Make Prediction")
    
    # Generate input fields based on model's input shape
    input_shape = model.input_shape[1:]  # Remove batch dimension
    num_features = input_shape[0] if isinstance(input_shape, tuple) else 1

    inputs = []
    cols = st.columns(3)
    for i in range(num_features):
        with cols[i % 3]:
            inputs.append(st.number_input(
                f"Feature {i+1}", 
                value=0.0,
                format="%.4f",
                step=0.0001
            ))

    if st.button("Predict"):
        try:
            # Prepare input data
            input_data = np.array([inputs]).astype(np.float32)
            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Display results
            st.subheader("Prediction Result")
            if model.output_shape[1] == 1:
                # Regression output
                st.metric("Predicted Value", f"{prediction[0][0]:.4f}")
            else:
                # Classification output
                classes = np.argmax(prediction, axis=1)
                st.write("Class Probabilities:")
                st.bar_chart(prediction[0])
                st.metric("Predicted Class", classes[0])
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

    # Batch prediction section
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Input Data Preview:", df.head())

            if st.button("Run Batch Prediction"):
                with st.spinner("Processing..."):
                    batch_input = df.values.astype(np.float32)
                    predictions = model.predict(batch_input)
                    
                    # Add predictions to dataframe
                    if model.output_shape[1] == 1:
                        df['prediction'] = predictions.flatten()
                    else:
                        df['predicted_class'] = np.argmax(predictions, axis=1)
                    
                    st.write("Predictions:", df)
                    
                    # Download results
                    st.download_button(
                        "Download Predictions",
                        df.to_csv(index=False),
                        "predictions.csv",
                        "text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Batch processing failed: {str(e)}")

if __name__ == "__main__":
    main()