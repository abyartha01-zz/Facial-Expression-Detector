# %% Importing libraries
from keras.preprocessing.image import img_to_array                              #to convert the read face to an array
import imutils                                                                  #to use the imutils.resize function to resize 
                                                                                #an image        
import cv2                                                                      #for working with images
from keras.models import load_model                                             #to load the required model 
import numpy as np                                                              #numpy is needed to work with arrays

# %% Creating necessary variables
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')      #loading the face detector    

emotion_model_path = "The name of the model you want"                           #loading the trained model
emotion_classifier = load_model(emotion_model_path, compile=False)

EMOTIONS = ["Angry", "Disgust", "Scared", "Happy", "Sad", 
            "Surprised", "Neutral"]

# %% Tracking face and predicting emotion
cv2.namedWindow('My_face')
camera = cv2.VideoCapture(0)

while True:
    frame = camera.read()[1]
    
    #reading the frame
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame,width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
#    print(faces)
    if len(faces) > 0:                                                          # Extract the ROI of the face from the grayscale                               
        faces = sorted(faces, reverse=True,                                     #image, resize it to a fixed 48x48 pixels, and then prepare        
                       key = lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]        # the ROI for classification via the CNN
        (fX, fY, fW, fH) = faces                                      
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48,48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
             
        preds = emotion_classifier.predict(roi)[0]                              #predicting the emotion of the face read        
        emotion_probability = np.max(preds)                                     #by the camera    
        label = EMOTIONS[preds.argmax()]
        
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):            #creating the rectangle bounding the face
            text = "{}: {:.2f}%".format(emotion, prob * 100)                    #and showing the emotion and also creating         
            w = int(prob * 300)                                                 #the window which shows the probabilities of     
            cv2.rectangle(canvas, (7, (i * 35) + 5),                            #all the emotions    
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
            cv2.putText(frameClone, label, (fX,fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, 
                        (0, 255, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (0, 155, 255), 1) 
                
        cv2.imshow('My_face', frameClone)                                       #displaying the face
        cv2.imshow("Probabilities", canvas)                                     #displaying the probabilit window     
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
