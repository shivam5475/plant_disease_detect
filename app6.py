import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.utils import img_to_array
from reponse4 import resp

def model_prediction(test_image):
            try:
                model = tf.keras.models.load_model("trained_plant_disease_model.keras")
                img = Image.open(test_image)
                img = img.resize((128, 128))
                input_arr = img_to_array(img)
                input_arr = np.array([input_arr])
                predictions = model.predict(input_arr)
                return np.argmax(predictions)
            except Exception as e:
                st.error("Error occurred during prediction: {}".format(e))
                return None

st.header("Disease Recognition")
st.write("Upload an image of a plant to detect diseases")
st.write("")
test_image = st.file_uploader("Choose an Image:")
if st.button("Show Image"):
    st.image(test_image, width=4, use_column_width=True)

    # Create a button to predict the disease
if st.button("Predict"):
    result_index = model_prediction(test_image)

            # Display the result
    if result_index is not None:
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
           # st.balloons()
        output=class_name[result_index]
        out=resp(output)
        st.write(out)
        
  