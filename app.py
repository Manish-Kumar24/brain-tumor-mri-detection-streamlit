import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import imutils
from PIL import Image
import io

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.keras')

def crop_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_thresh = cv2.threshold(img_blur, 45, 255, cv2.THRESH_BINARY)[1]
    img_thresh = cv2.erode(img_thresh, None, iterations=2)
    img_thresh = cv2.dilate(img_thresh, None, iterations=2)
    
    contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    
    if not contours:
        return image
    
    c = max(contours, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()])[0]
    extRight = tuple(c[c[:, :, 0].argmax()])[0]
    extTop = tuple(c[c[:, :, 1].argmin()])[0]
    extBottom = tuple(c[c[:, :, 1].argmax()])[0]
    
    new_img = image[extTop[1]:extBottom[1], extLeft[0]:extRight[0]]
    return new_img

def preprocess_image(upload):
    # Convert uploaded file to image
    image = Image.open(upload)
    image = np.array(image)
    
    # Convert RGB to BGR if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Crop and resize
    image = crop_image(image)
    image = cv2.resize(image, (240, 240))
    
    # Convert back to RGB for display
    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare for model
    image = np.expand_dims(image, axis=0)
    return image, display_image

def main():
    st.title("Brain Tumor MRI Detection")
    st.write("Upload a brain MRI image for tumor detection")
    
    try:
        model = load_model()
    except Exception as e:
        st.error("Error loading model. Please ensure 'model.keras' is in the same directory.")
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with st.spinner('Processing image...'):
            # Preprocess image
            image, display_image = preprocess_image(uploaded_file)
            
            # Display original image
            with col1:
                st.subheader("Uploaded Image")
                st.image(display_image, use_column_width=True)
            
            # Make prediction
            prediction = model.predict(image)
            class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            predicted_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction)) * 100
            
            # Display results
            with col2:
                st.subheader("Prediction Results")
                st.write(f"**Diagnosis:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                
                # Display probability bars
                st.write("\n**Probability Distribution:**")
                for i, (class_name, prob) in enumerate(zip(class_names, prediction[0])):
                    st.progress(float(prob))
                    st.write(f"{class_name}: {float(prob)*100:.2f}%")

if __name__ == '__main__':
    main()