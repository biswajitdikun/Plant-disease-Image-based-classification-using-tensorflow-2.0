#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals


# In[3]:


# !pip install tensorflow-gpu==2.0.0


# In[2]:


import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import os
from tensorflow import keras
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


# import pathlib
# data_dir = "/content/drive/My Drive/plant_disease"
# DATADIR = pathlib.Path(data_dir)
# train_dir="/content/drive/My Drive/plant_disease/train"
# test_dir="/content/drive/My Drive/plant_disease/test"
# TRAINDIR = pathlib.Path(train_dir)
# TESTDIR = pathlib.Path(test_dir)


# In[3]:


import pathlib
data_dir = "C:/Users/Biswajit Satapathy/Desktop/study/plant_disease"
DATADIR = pathlib.Path(data_dir)
train_dir="C:/Users/Biswajit Satapathy/Desktop/study/plant_disease/train"
test_dir="C:/Users/Biswajit Satapathy/Desktop/study/plant_disease/test"
TRAINDIR = pathlib.Path(train_dir)
TESTDIR = pathlib.Path(test_dir)


# In[6]:


CATEGORIES = ["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy","Blueberry___healthy","Cherry_(including_sour)___healthy","Cherry_(including_sour)___Powdery_mildew","Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___healthy", "Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight", "Potato___healthy", "Potato___Late_blight","Raspberry___healthy","Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot","Tomato___Tomato_mosaic_virus","Tomato___Tomato_Yellow_Leaf_Curl_Virus"] 


# In[5]:


# CATEGORIES = ["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy","Blueberry___healthy","Cherry_(including_sour)___healthy","Cherry_(including_sour)___Powdery_mildew","Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___healthy", "Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight", "Potato___healthy", "Potato___Late_blight","Raspberry___healthy","Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot","Tomato___Tomato_mosaic_virus","Tomato___Tomato_Yellow_Leaf_Curl_Virus"] 


# In[104]:


len(CATEGORIES)


# In[105]:


for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(TRAINDIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        #plt.imshow(img_array)  # graph it
        #plt.show()  # display!

        break  # we just want one for now so break
    break
for category in CATEGORIES:  # do dogs and cats
    path1 = os.path.join(TESTDIR,category)  # create path to dogs and cats
    for img1 in os.listdir(path1):  # iterate over each image per dogs and cats
        img_array1 = cv2.imread(os.path.join(path1,img1) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        #plt.imshow(img_array1, cmap='gray')  # graph it
        #plt.show()  # display!

        break  # we just want one for now so break
    break      


# In[106]:


from tqdm import tqdm


# In[107]:


train_image_generator = ImageDataGenerator(rescale=1./255,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           rotation_range=45,
                                           width_shift_range=.15,
                                           height_shift_range=.15,
                                           zoom_range=0.5
                                           ) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           rotation_range=45,
                                           width_shift_range=.15,
                                           height_shift_range=.15,
                                           zoom_range=0.5
                                           ) # Generator for our validation data


# In[108]:


batch_size = 16
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


# In[109]:


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=TRAINDIR,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')


# In[110]:


val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=TESTDIR,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')


# In[111]:


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),   
    Dropout(0.5),
    Dense(512, activation='relu'),
    Flatten(),
    Dense(34, activation='softmax')
])


# In[93]:


# model = Sequential([
#     Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),  
#     Conv2D(32, 3, padding='same', activation='relu'),
#     Conv2D(32, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Dropout(0.25),
#     Conv2D(64, 3, padding='same', activation='relu'),
#     Conv2D(64, 3, padding='same', activation='relu'),
#     MaxPooling2D(),   
#     Dropout(0.25),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.50),
#     Dense(11, activation='softmax')
# ])


# In[112]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[113]:


model.summary()


# In[114]:


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=2007 // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=206 // batch_size
)


# In[115]:


acc = history.history['accuracy']
print(acc)
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[4]:


model.save('plant_disease.h5')


# In[2]:


import cv2
import numpy as np


# In[3]:


from tensorflow.keras import models,layers,optimizers


# In[4]:


model1=models.load_model('plant_disease.h5',compile=True)


# In[2]:


# model1.hist()


# In[4]:


img3 = cv2.imread(r"C:\Users\Biswajit Satapathy\Desktop\study\pe.jpg",1)
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
img3 = cv2.resize(img3,(150,150))
print(img3.shape)
img4 = np.reshape(img3,[1,150,150,3])
img4=img4/255.0


# In[7]:


disease = model1.predict_classes(img4)
prediction = disease[0]
print(CATEGORIES[prediction])

