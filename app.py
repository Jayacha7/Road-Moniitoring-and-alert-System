import streamlit as st
import os
from os import listdir
import cv2     # for capturing videos
import math 
import geocoder
from PIL import Image, ImageOps
import numpy as np
import tempfile
import tensorflow as tf
import pandas as pd
from twilio.rest import Client as client
from geopy.geocoders import Nominatim
from keras.preprocessing import image   # for preprocessing the images
from keras.utils import np_utils
from matplotlib import pyplot as plt 
from keras.layers import Dense, InputLayer, Dropout
from keras.models import Sequential
from skimage.transform import resize
import numpy as np

st.title("Accident Detection")

def load_model():
  model=tf.keras.models.load_model('model-new.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

def save_frames_to_image(path, video_file):
    count = 1
    cap = cv2.VideoCapture(video_file)
    frameRate = cap.get(5) #frame rate
    while(cap.isOpened()):
        frameId = cap.get(1)
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename ="%d.jpg" % count;count+=1
            cv2.imwrite(os.path.join(path ,filename), frame)
    cap.release()
    st.write("Done!")

def import_and_predict(image_data, model):
    
        size = (250,250)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction  

# @st.cache(allow_output_mutation=True)
# # st.title("Save Video Frames to Images")
# # st.sidebar.header('User Input')
# @st.cache(suppress_st_warning=True)


#for messaging dataflow
geoLoc = Nominatim(user_agent="GetLoc")
g = geocoder.ip('me')
locname = geoLoc.reverse(g.latlng)
account_sid =  'AC9471a76c17ed75eb9b88eb209a9d61cd'#Enter Your account sid
auth_token ='b788c013b9aa4f14b5183fb3c7f57307' #Enter your auth token
clientq = client(account_sid, auth_token)

msg = 0

path = "./temp"
class_names= ['accident','not-accident']
model=load_model()
count =0
predictions=[]
i=0
res=0
videoFile = st.file_uploader("Upload video file")
for file in os.listdir(path):
    if file.endswith('.jpg'):
        os.remove(os.path.join(path,file)) 
# st.set_option('deprecation.showfileUploaderEncoding', False)
if videoFile is None:
        st.text("Please upload an video file")
else:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(videoFile.read())
    save_frames_to_image(path,tfile.name)
    for image in os.listdir(path):
        image = Image.open(os.path.join(path,image))
        predictions.append(import_and_predict(image, model))
        score = tf.nn.softmax(predictions[i][0])
        i+=1
        # st.write(predictions[i][0])
        if count>1  :
            st.image(image, use_column_width=True)
            st.write(score)
            # string = "{}  {:.2f} percent ".format(class_names[np.argmax(score)], 100 * np.max(score))
            location = locname.address
            string = "Accident has occured in {}".format(location)
            msg = 1
            # clientq.messages.create(
            #      body="Accident detected in "+locname.address,
            #      from_= '+16203776457',
            #      to= '+916382906223'
            #     )  
            st.text(string) 
        elif(class_names[np.argmax(score)]=="accident"):
            count+=1
            continue
        else:
            res=count
            count=0
    if msg:
        clientq.messages.create(
            body="Accident detected in "+locname.address,
            from_= '+16203776457',
            to= '+916382906223'
            )  