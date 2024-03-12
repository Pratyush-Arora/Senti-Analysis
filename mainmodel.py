import cv2
from keras.models import model_from_json
import numpy as np
import sys

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Suppress UnicodeEncodeError by redirecting stdout
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='UTF-8', buffering=1)

webcam = cv2.VideoCapture(0)
# Add a small delay for webcam initialization
cv2.waitKey(1000)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

try:
    while True:
        i, im = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(im, 1.3, 5)

        try:
            for (p, q, r, s) in faces:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(im, (p, q), (p+r, q+s), (0, 255, 0), 2)  # Change rectangle color to green
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                
                # Display the predicted emotion label on the output window
                cv2.putText(im, prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
                
            cv2.imshow("Output", im)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        except cv2.error:
            pass
except KeyboardInterrupt:
    print("Interrupted, closing the application.")
finally:
    webcam.release()
    cv2.destroyAllWindows()
