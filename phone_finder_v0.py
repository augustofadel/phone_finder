#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phone Finder V0

@author: augustofadel
"""

#from os import listdir
from os.path import join
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

#from tensorflow.contrib.keras.python import keras
#from tensorflow.contrib.keras.python.keras.models import Sequential
#from tensorflow.contrib.keras.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
#from tensorflow.contrib.keras.python.keras.preprocessing.image import load_img, img_to_array



#image_path = 'D:/Users/augusto.fadel/Documents/!filesync/mestrado/18.1_aprendizado de maquina/trabalho2/find_phone_data'
#image_path = '/media/augusto/AUGUSTO/mestrado/18.1_aprendizado de maquina/trabalho2/find_phone_data'
#image_path = '/home/augusto/Desktop/filesync/mestrado/18.1_aprendizado de maquina/trabalho2/find_phone_data'
image_path = '/home/augustofadel/phone_finder/find_phone_data'

phone_location = pd.read_csv(join(image_path, 'labels.txt'),
                             sep = ' ',
                             header = None,
                             names = ['image_name', 'x_coord', 'y_coord'])

image_height = 128 #128 #163 #326
image_width = 192 #192 #245 #490
images = [load_img(join(image_path, img), 
                   target_size = (image_height, image_width)) 
          for img in phone_location['image_name']]

#i = 10
#from IPython.display import display
#print(display(images[i]))
#print(phone_location.loc[[i]])

X = np.asarray([np.array(img_to_array(img)) for img in images])/255
y = np.asarray(phone_location.drop('image_name', axis = 1))



model = Sequential()
model.add(Conv2D(64, 
                 kernel_size = (3, 3),
                 #use_bias = True,
                 activation = 'relu',
                 input_shape = (image_height, image_width, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, 
                 kernel_size = (3, 3),
                 #use_bias = True,
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, 
                 kernel_size = (3, 3),
                 #use_bias = True,
                 activation = 'relu'))
model.add(Flatten())
model.add(Dense(128, 
                activation = 'relu'))
model.add(Dense(2,
                activation = 'softmax'))

model.compile(loss = keras.losses.mean_squared_error,
              optimizer = 'adam',
              metrics = ['mae', 'mse'])

model.fit(X,
          y,
          batch_size = 10,
          epochs = 3,
          validation_split = 0.2)
