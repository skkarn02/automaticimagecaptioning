
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle as pkl


#Loading pretrained Resnet model 
model=ResNet50(weights="imagenet",input_shape=(224,224,3))
model_new=Model(model.input,model.layers[-2].output)
#Python function to preprocess input 

def preprocess_image(image_file) :
    img_arr=cv2.resize(image_file,(224,224))
    img_arr=np.expand_dims(img_arr,axis=0)
    img_arr=preprocess_input(img_arr)
    
    feature_vector=model_new.predict(img_arr)
    feature_vector=feature_vector.reshape((2048,))
    
    return feature_vector


#Load wordto index and index to word dictionaries 
with open("word_to_idx.pkl", "rb") as input_file:
    word_to_idx = pkl.load(input_file)
    
with open("idx_to_word.pkl", "rb") as input_file:
    idx_to_word = pkl.load(input_file)
    
    
#Load model 
#ic_model=keras.models.load_model('model_80.h5')
ic_model=keras.models.load_model('model_801.h5') 
#Predict function 
def predict_caption(photo):
    in_text = 'startseq'
    for i in range(35):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=35)
        yhat = ic_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final
      

st.title("Automatic Image Captioning")
st.header("Upload a image to get a neural caption for it")
# To View Uploaded Image
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
# You don't have handy image 
if bool(image_file)==True :
    img=plt.imread(image_file)
    img_arr=cv2.resize(img,(256,256))
    st.image(img_arr)
    fv=preprocess_image(img_arr)
    fv=fv.reshape((1,2048))
    st.write(predict_caption(fv))
else :
    ran_imageid=['416960865_048fd3f294','1392272228_cf104086e6','2599444370_9e40103027','3384314832_dffc944152','3482974845_db4f16befa','3558370311_5734a15890','542179694_e170e9e465','2739331794_4ae78f69a0','2886411666_72d8b12ce4']
    st.markdown("OOPS !!!!!!!!!! You are not ready with some images 😬. Don't worry i have some images for you click on the below button and it will generate caption to a random image from a set of images. 😎")
    if st.button('Generate Caption for a random image') :
        ran_num=np.random.randint(0,9)
        img_static_path=str(ran_imageid[ran_num])+'.jpg'
        img_static=plt.imread(img_static_path)
        img_static_arr=cv2.resize(img_static,(256,256))
        st.image(img_static_arr)
        fvs=preprocess_image(img_static_arr)
        fvs=fvs.reshape((1,2048))
        st.write('##### PREDICTED CAPTION')
        st.write(predict_caption(fvs))
    
    
