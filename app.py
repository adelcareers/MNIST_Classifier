import streamlit as st
import numpy as np
from PIL import Image
import torch
import pandas as pd
from inference import MNISTInference
from db_logger import log_prediction, init_db, get_prediction_history
from streamlit_drawable_canvas import st_canvas

def main():
    # Initialize database on startup
    init_db()
    
    st.title("MNIST Digit Classifier")
    
    # Initialize the MNIST classifier
    mnist_classifier = MNISTInference()
    
    # Create canvas for drawing
    canvas = st_canvas(
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    
    # Create columns for input and prediction
    col1, col2 = st.columns(2)
    
    # Add true label input in the first column
    with col1:
        true_label = st.number_input("True Label:", 
                                    min_value=0, 
                                    max_value=9, 
                                    step=1)
    
    # Add predict button in the second column
    with col2:
        predict_button = st.button('Predict')
    
    if canvas.image_data is not None and predict_button:
        # Convert canvas to numpy array
        image = Image.fromarray(canvas.image_data.astype('uint8'))
        image = image.convert('L')  # Convert to grayscale
        
        # Get prediction
        prediction, confidence = mnist_classifier.predict(image)
        
        # Display results
        st.write("### Results:")
        st.write(f"Predicted Digit: {prediction}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.write(f"True Label: {true_label}")
        
        # Log prediction
        log_prediction(prediction, true_label, confidence)
        
        # Display prediction history
        st.write("### History:")
        history = get_prediction_history()
        if history:
            df = pd.DataFrame(history, columns=['Timestamp', 'Prediction', 'True Label'])
            st.dataframe(df)
        else:
            st.write("No prediction history available.")

if __name__ == "__main__":
    main()
