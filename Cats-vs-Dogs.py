
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Conv2D,MaxPooling2D, Flatten, Input
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

dir = ''  # Directory path for your Training data

X = []
y = []
for filename in os.listdir(dir):
    imgPath = os.path.join(dir, filename)
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (100,100))
    label = filename.split('.')[0]
    X.append(img)
    y.append(label)
    
X = np.array(X)
y = np.array(y)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train = X_train/255
# X_test = X_test/255
# y_train = np.array(y_train)


model = Sequential([
    Input(shape=(100, 100, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

X = X/255

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,y,epochs=10,batch_size=64)
model.evaluate(X_test,y_test)



