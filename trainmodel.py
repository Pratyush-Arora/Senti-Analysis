from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np
import sys




TRAIN_DIR = 'images/train'
TEST_DIR = 'images/test'


def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths,labels


train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)




print(train)




test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)




from tqdm.notebook import tqdm




import numpy as np
import cv2
from tqdm import tqdm

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        features.append(img)
    features = np.array(features).reshape(len(features), 48, 48, 1)
    return features


train_features = extract_features(train['image']) 



test_features = extract_features(test['image'])



x_train = train_features/255.0
x_test = test_features/255.0



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()
le.fit(train['label'])




y_train = le.transform(train['label'])
y_test = le.transform(test['label'])




y_train = to_categorical(y_train,num_classes = 7)
y_test = to_categorical(y_test,num_classes = 7)

if sys.stdout.encoding != 'UTF-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='UTF-8', buffering=1)


from keras.layers import Input

# Define your model
model = Sequential()
model.add(Input(shape=(48, 48, 1)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))

# Save the model
model_json = model.to_json()
with open("emotiondetector.json", 'w') as json_file:
    json_file.write(model_json)
model.save("emotiondetector.h5")