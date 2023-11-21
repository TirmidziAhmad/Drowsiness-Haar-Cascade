import cv2
import tensorflow
from tensorflow import keras
import numpy as np
import time
from pygame import mixer
#import serial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                     + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                    + 'haarcascade_eye.xml')
model = keras.models.load_model(r'C:\Users\tirmi\Downloads\Drowsiness-Haar-Cascade\models\model.h5')

mixer.init()
sound= mixer.Sound(r'C:\Users\tirmi\Downloads\Drowsiness-Haar-Cascade\sound\alarm.wav')
cap = cv2.VideoCapture(0)
Score = 0
Kedip = 0
while True:
    ret, frame = cap.read()
    height,width = frame.shape[0:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, scaleFactor= 1.2, minNeighbors=3)
    eyes= eye_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=1)
    
    cv2.rectangle(frame, (0,height-50),(290,height),(0,0,0),thickness=cv2.FILLED)
    cv2.rectangle(frame, (0,height-120),(290,height),(0,0,0),thickness=cv2.FILLED)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h), color= (255,0,0), thickness=3 )
        
    for (ex,ey,ew,eh) in eyes:
        #cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color= (0,0,255), thickness=3 )
        
        # preprocessing steps
        eye= frame[ey:ey+eh,ex:ex+ew]
        eye= cv2.resize(eye,(80,80))
        eye= eye/255
        eye= eye.reshape(80,80,3)
        eye= np.expand_dims(eye,axis=0)
        # preprocessing is done now model prediction
        prediction = model.predict(eye)
        print(prediction)
        
        # if eyes are closed
        if prediction[0]<0.50:
            Score = Score+1
            cv2.putText(frame,'Tertutup',(30,height-20),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            cv2.putText(frame,str(Score)+' Detik',(170,height-20),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            if(Score>4):
                cv2.putText(frame,'Mengantuk!',(30,height-90),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1,color=(0,0,255),
                       thickness=1,lineType=cv2.LINE_AA)
                time.sleep(2)
                try:
                    sound.play()
                except:
                    pass
               
            
        # if eyes are open
        elif prediction[0]>0.80:
            cv2.putText(frame,'Mata Terbuka',(30,height-50),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=1,color=(0,255,0),
                       thickness=1,lineType=cv2.LINE_AA)
            time.sleep(2)

            Score = Score-1
            if (Score<0):
                Score=0
            
        
    cv2.imshow('Deteksi Kantuk',frame)
    if cv2.waitKey(33) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()