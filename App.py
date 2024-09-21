import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Ensemble_functions import output_label,ensemble_output,image_preprocess,load_models
from PIL import Image
import gdown

st.title('🧠 Brain Tumour Image Classification')

st.subheader("Description")
st.write("This is mult-label image classification model, Given an image the model classifies it into\n\n"\
         "\n1. Glioma\n2. Meningioma\n 3. No Tumour\n 4. Pituitary")


st.divider()
# Option to download model data images
st.subheader('Download the images the model has trained on')


gdown.download(id = "11R-D1robYjdyMZ4KooWuAlZH1WyRpCdl=",output = "BrainTumor_1.zip")
with open("BrainTumor_1.zip","rb") as f:
         st.download_button(label='download data',
                           data = f,
                           mime='application/zip')


st.divider()
st.subheader('Data Visualization')
on = st.toggle("View data distribution")
data_df = pd.DataFrame(data={'Glioma':5284,'Meningioma':5356,"No Tumour":5828,"Pituitary":6380},index=[0])

if on:
    st.bar_chart(data=data_df,color=['#06C','#4CB140','#009596','#F0AB00'],stack=False)

st.subheader('Upload the image')
file = st.file_uploader(label='Image file',
                 label_visibility='hidden'
                 ,type=['png', 'jpg','jpeg'])

model_input = 0
if file is not None:
    image = Image.open(file)
    image_array = np.array(image)
    preprocessed_img = image_preprocess(image_array)
    preprocessed_img_np = preprocessed_img.numpy()
    model_input = preprocessed_img_np
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratios if needed

    with col2:  # Center column
        st.image(preprocessed_img_np, width=350)


# click = st.button('Predict')
# output_arr = 0
# class_name_predicted = ""
# if click:
#     ## Loading the models
#     loading_flag = 0
#     with st.spinner('Loading models....'):
#         densenet,vgg19,xception,effnet = load_models()
#         loading_flag = 1
#     st.success('Models loaded successfully')
    
#     st.subheader('Output')
#     with st.spinner('Predicting.....'):
#         output_arr,class_name_predicted = ensemble_output(model_input,densenet,vgg19,xception,effnet)
#     st.progress(output_arr[0],text='Glioma')
#     st.progress(output_arr[1],text='Meningioma')
#     st.progress(output_arr[2],text='No Tumour')
#     st.progress(output_arr[3],text='Pituitary')
densenet_url = "https://drive.google.com/uc?id=1alRU89gEjm1hc1TJZ965Sg40gJrXap5g"
vgg19_url = "https://drive.google.com/uc?id=1E_qVWwNkDj-vbYO0Rlx4JoexCxGtIw9_"
xception_url = "https://drive.google.com/uc?id=1YMo2BkbuqCwoRi6-XfT0P5SIWyf82VEE"
effnet_url = "https://drive.google.com/uc?id=1xsk9pUCAQuztZyaa5UJwAq4cwxChUIfl"

with st.spinner('Getting the models ready.....'):
         gdown.download(densenet_url,"densenet169_model.keras")
         gdown.download(vgg19_url,"VGG19_model.keras")
         gdown.download(xception_url,"xception_model.keras")
         gdown.download(effnet_url,"EfficientNetV2B2_model.keras")
         densenet = tf.keras.models.load_model("densenet169_model.keras")
         vgg19 = tf.keras.models.load_model("VGG19_model.keras")
         xception = tf.keras.models.load_model("xception_model.keras")
         effnet = tf.keras.models.load_model("EfficientNetV2B2_model.keras")
         
