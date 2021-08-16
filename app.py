import numpy as np
import cv2
from keras.models import load_model 
import streamlit as st


model_path = 'trained_model/model.h5'
emotion_model = load_model(model_path)

st.title("Face to Emoji")

uploaded_image = st.file_uploader("Upload your image and see magic", ["jpg","jpeg","png"])
image = None
maxindex = None

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

if image is not None:
    frame = image

    bounding_box = cv2.CascadeClassifier('trained_model/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    if not maxindex: 
        st.warning('Enable to detect emotion')
    
    
    else: 
        st.image(cv2.resize(frame,(450,450),interpolation = cv2.INTER_CUBIC))
        emoji = cv2.imread('emojis/' + emotion_dict[maxindex].lower() + '.png')
        st.image(cv2.resize(emoji,(450,500),interpolation = cv2.INTER_CUBIC))

