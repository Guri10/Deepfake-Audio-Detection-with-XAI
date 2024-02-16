import streamlit as st
import pandas as pd
import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model
import librosa.display, os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model
import tensorflow as tf
from keras.applications.mobilenet import preprocess_input
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.util import img_as_float
import cv2 
import time

st.set_page_config(page_title="Deepfake Audio Detection",page_icon="")

class_names = ['real','fake']

def save_file(sound_file):
    # save your sound file in the right folder by following the path
    with open(os.path.join('audio_files/', sound_file.name),'wb') as f:
         f.write(sound_file.getbuffer())
    return sound_file.name

def create_spectrogram(sound):
    audio_file = os.path.join('audio_files/', sound)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    # st.pyplot(fig)
    plt.savefig('melspectrogram.png')
    image_data = load_img('melspectrogram.png',target_size=(224,224))
    st.image(image_data)
    return(image_data)

def predictions(image_data,model):
  
#   model.summary(print_fn=lambda x: st.text(x))

#   img_array = img_to_array(image_data)
#   img_batch = np.expand_dims(img_array, axis=0)

#   img_preprocessed = preprocess_input(img_batch)
#   prediction = model.predict(img_preprocessed)

#   class_label = np.argmax(prediction)
    img_array = np.array(image_data)
    img_array1 = img_array / 255
    img_batch = np.expand_dims(img_array1, axis=0)

    # img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_batch)
    class_label = np.argmax(prediction)
    return class_label,prediction

def lime_predict(image_data,model):
    img_array = np.array(image_data)
    img_array1 = img_array / 255
    img_batch = np.expand_dims(img_array1, axis=0)

    # img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_batch)
    class_label = np.argmax(prediction)

    explainer = lime.lime_image.LimeImageExplainer()
    # explanation = explainer.explain_instance(img_array.astype('float64'), model.predict, hide_color=0, num_samples=1000)
    explanation = explainer.explain_instance(img_array1.astype('float64'), model.predict, hide_color=0, num_samples=1000)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 25))
    for i in range(2):
        # Show the original image and the explanation
        temp, mask = explanation.get_image_and_mask(np.argmax(prediction[0], axis=0), positive_only=False, num_features=8, hide_rest=True)
        axs[0].imshow(image_data)
        axs[1].imshow(mark_boundaries(temp, mask))
        axs[1].set_title(f"Predicted class: {class_names[class_label]}")
    plt.tight_layout()
    # plt.show()
    # plt.savefig('XAI_output.png')
    st.pyplot(fig)
    return(fig)

def grad_predict(image_data,model_mob,preds,class_idx):
    img_array = img_to_array(image_data)
    # img_array1 = img_array / 255
    x = np.expand_dims(img_array,axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)

    model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    last_conv_layer = model.get_layer('block5_conv3')
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(x)
        class_output = preds[:, class_idx]
    grads = tape.gradient(class_output, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    heatmap = cv2.resize(np.float32(heatmap), (x.shape[2], x.shape[1]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32)
    superimposed_img = cv2.addWeighted(x[0], 0.6, heatmap, 0.4, 0, dtype = cv2.CV_32F)

    # fig, ax = plt.subplots()
    # ax.title('Grad-CAM visualization')
    # st.write(superimposed_img)
    # plt.imshow(superimposed_img)
    # plt.savefig('XAI_output.png')
    # st.pyplot(fig)
    # st.pyplot(superimposed_img)

    fig1, ax = plt.subplots(1, 2, figsize=(10, 25))
    for i in range(2):
        # Show the original image and the explanation
        ax[0].imshow(image_data)
        ax[1].imshow(superimposed_img)
        ax[1].set_title(f"Predicted class: {class_names[class_idx]}")
    plt.tight_layout()
    # plt.show()
    # plt.savefig('XAI_output.png')
    st.pyplot(fig1)
    return(superimposed_img)

def main():
    page = st.sidebar.selectbox("App Selections", ["Homepage", "About"])
    if page == "Homepage":
        st.title("Deepfake Audio Detection using XAI")
        homepage()
    elif page == "About":
        about()

def about():
    # st.set_page_config(layout="centered")
    st.title("About present work")
    st.markdown("**Deepfake audio refers to synthetically created audio by digital or manual means. An emerging field, it is used to not only create legal digital hoaxes, but also fool humans into believing it is a human speaking to them. Through this project, we create our own deep faked audio using Generative Adversarial Neural Networks (GANs) and objectively evaluate generator quality using Fréchet Audio Distance (FAD) metric. We augment a pre-existing dataset of real audio samples with our fake generated samples and classify data as real or fake using MobileNet, Inception, VGG and custom CNN models. MobileNet is the best performing model with an accuracy of 91.5% and precision of 0.507. We further convert our black box deep learning models into white box models, by using explainable AI (XAI) models. We quantitatively evaluate the classification of a MEL Spectrogram through LIME, SHAP and GradCAM models. We compare the features of a spectrogram that an XAI model focuses on to provide a qualitative analysis of frequency distribution in spectrograms.**")
    st.markdown("**The goal of this project is to study features of audio and bridge the gap of explain ability in deep fake audio detection, through our novel system pipeline. The findings of this study are applicable to the fields of phishing audio calls and digital mimicry detection on video streaming platforms. The use of XAI will provide end-users a clear picture of frequencies in audio that are flagged as fake, enabling them to make better decisions in generation of fake samples through GANs.**")

def homepage():
    st.write('___')
    st.subheader("Choose a wav file")
    uploaded_file = st.file_uploader(' ', type='wav')
    if uploaded_file is not None:  
        # view details
        # file_details = {'filename':uploaded_file.name, 'filetype':uploaded_file.type, 'filesize':uploaded_file.size}
        # st.write(file_details)
        # read and play the audio file
        st.write('### Play audio')
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')

        st.write('### Spectrogram Image:')
        save_file(uploaded_file)
        # define the filename
        sound = uploaded_file.name
        with st.spinner('Fetching Results...'):
            spec = create_spectrogram(sound)
            model = tf.keras.models.load_model('saved_model/model')
        st.write('### Classification results:')
        class_label,prediction = predictions(spec,model)
        st.write("#### The uploaded audio file is "+class_names[class_label])
        if st.button('Show XAI Metrics'):
            st.write('### XAI Metrics using Lime ')
            with st.spinner('Fetching Results...'):
                fig2 = lime_predict(spec,model)
            st.write('### XAI Metrics using Grad CAM ')
            with st.spinner('Fetching Results...'):
                grad_img = grad_predict(spec,model,prediction,class_label)
    elif uploaded_file is None:
        st.info("Please upload an .wav file")


if __name__ == "__main__":
    main()