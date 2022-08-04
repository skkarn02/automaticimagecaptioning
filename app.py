
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50 , preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle as pkl
from PIL import Image


#Loading pretrained Resnet model 
#inception = InceptionV3(weights='imagenet',input_shape=(299,299,3))
rm=ResNet50(weights="imagenet",input_shape=(224,224,3))
rm_new=Model(rm.input,rm.layers[-2].output)
#im=Model(inception.input,inception.layers[-2].output)
#Python function to preprocess input 

def preprocess_image(img) :
    img1=Image.open(img)
    img = img1.resize((224,224),Image.ANTIALIAS)
    img=np.expand_dims(img,axis=0)
    #img=preprocess_input(img)
    feature_vector = rm_new.predict(img)
    feature_vector = feature_vector.reshape((2048,))
    return feature_vector


#Load wordto index and index to word dictionaries 
with open("word_to_idx.pkl", "rb") as input_file:
    word_to_idx = pkl.load(input_file)
    
with open("idx_to_word.pkl", "rb") as input_file:
    idx_to_word = pkl.load(input_file)
    
    
#Load model 
ic_model=load_model('model_final.h5')
#ic_model=keras.models.load_model('model_801.h5') 
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
    img_temp=plt.imread(image_file)
    st.image(img_temp)
    fv=preprocess_image(image_file)
    fv=fv.reshape((1,2048))
    caption=predict_caption(fv)
    st.write('##### PREDICTED CAPTION')
    st.write(caption)
    st.text("")
    st.text("")
    st.text("")        
    st.write("##### NOTE : This prediction is based on basic encoder-decoder model . I am currently working on it to improve it using advance encoder-decoder architecture such as Attention Models and Transformer . Please Stay Tuned . Thankyou ❤️")
        
else :
    ran_imageid=['1298295313_db1f4c6522','3203453897_6317aac6ff','3482974845_db4f16befa','3655155990_b0e201dd3c','3558370311_5734a15890']
    st.text("")
    st.text("")
    st.text("You can download some sample images by clicking on the below links :")
    st.write("[link](https://drive.google.com/file/d/1IDVaVABqEr5IyiM1MoDAOdUIcvkSkKE_/view?usp=sharing)")
    st.write("[link](https://drive.google.com/file/d/1IDVaVABqEr5IyiM1MoDAOdUIcvkSkKE_/view?usp=sharing)")
    st.write("[link](https://drive.google.com/file/d/1IDVaVABqEr5IyiM1MoDAOdUIcvkSkKE_/view?usp=sharing)")
    st.text("")
    st.text("")
    st.markdown("OOPS !!!!!!!!!! You are not ready with some images 😬. Don't worry i have some images for you click on the below button and it will generate caption to a random image from a set of images. 😎")
    if st.button('Generate Caption for a random image') :
        ran_num=np.random.randint(0,len(ran_imageid))
        img_static_path=str(ran_imageid[ran_num])+'.jpg'
        img_temp=plt.imread(img_static_path)
        st.image(img_temp)
        fvs=preprocess_image(img_static_path)
        fvs=fvs.reshape((1,2048))
        st.write('##### PREDICTED CAPTION')
        st.write(predict_caption(fvs))
        st.text("")
        st.text("")
        st.text("")        
        st.write("##### NOTE : This prediction is based on basic encoder-decoder model. I am currently working on it to improve it using advance encoder-decoder architecture such as Attention Models and Transformer . Please Stay Tuned . Thankyou ❤️")
            


    