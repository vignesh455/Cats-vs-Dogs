#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.layers import Activation,Dense,Conv2D,MaxPooling2D, Flatten
from keras.models import Sequential


# In[2]:


dir = r'C:\Users\vigne\Desktop\V\datasets\Train'
dir1 = r'C:\Users\vigne\Desktop\V\datasets\Test'
types = ['Cat','Dog']


# In[3]:


Train = []
Test = []

for i in types:
    p = os.path.join(dir,i)
    c = types.index(i)
    for j in os.listdir(p):
        try:
            img = cv2.imread(os.path.join(p,j))
            img1 = cv2.resize(img,(100,100))
            Train.append([img1,c])
        except Exception as e:
            pass

for i in types:
    p = os.path.join(dir1,i)
    for j in os.listdir(p):
        try:
            img = cv2.imread(os.path.join(p,j))
            img = cv2.resize(img,(100,100))
            Test.append(img)
        except Exception as e:
            pass


# In[4]:


xtrain = []
ytrain = []

for i,j in Train:
    xtrain.append(i)
    ytrain.append(j)

xtrain = np.array(xtrain)/255
ytrain = np.array(ytrain)


# In[5]:


model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)),
    MaxPooling2D(2,2),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
])


# In[6]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[7]:


model.fit(xtrain,ytrain,epochs=10,batch_size=64)


# In[8]:


model.evaluate(xtrain,ytrain)


# In[36]:


r = random.randint(0,len(xtrain))
plt.imshow(xtrain[r])
plt.show()
pre = model.predict(xtrain[r].reshape(1,100,100,3))
print(pre)
if pre>0.5:
    print("Its a DOG")
elif pre<0.5:
    print("Its a CAT")


# In[ ]:




