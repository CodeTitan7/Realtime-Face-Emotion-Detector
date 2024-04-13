import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

os.chdir(os.path.dirname(__file__))
emotion_model = load_model("emotion_detection_model.keras")

emotion_labels = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]  
        resized_roi = cv2.resize(face_roi, (48, 48))  
        normalized_roi = resized_roi / 255.0  
        reshaped_roi = np.reshape(normalized_roi, (1, 48, 48, 1))  
        emotion_prediction = emotion_model.predict(reshaped_roi)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_label = emotion_labels[emotion_label_arg]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or cv2.getWindowProperty("Emotion Detection", cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()