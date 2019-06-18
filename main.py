# shuf -n 10 -e * | xargs -i mv {} path-to-new-folder

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
import numpy as np
import cv2
import os

nbatch = 64

train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=12., width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.15, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory('./datasets/fingers/train/', target_size=(224, 224), color_mode='rgb',
                                              batch_size=nbatch, shuffle=True,
                                              classes=['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                              class_mode='categorical')

test_gen = test_datagen.flow_from_directory('./datasets/fingers/test/', target_size=(224, 224), color_mode='rgb',
                                            batch_size=nbatch,
                                            classes=['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                            class_mode='categorical')

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_TEST = test_gen.n // test_gen.batch_size
print(STEP_SIZE_TEST, STEP_SIZE_TRAIN)

history = model.fit_generator(train_gen, steps_per_epoch=STEP_SIZE_TRAIN, epochs=10, validation_data=test_gen,
                              validation_steps=STEP_SIZE_TEST)

img = cv2.imread('./fingers/3R_test.png')
print(img.shape)
img = np.expand_dims(img, axis=0)
print(img.shape)

"""
>>> img = cv2.imread('./fingers/validation2/2R.jpg')
>>> img = cv2.resize(img, (128,128))
>>> img = np.expand_dims(img, axis=0)
>>> model.predict(img)

y_prob = model.predict(x) 
y_classes = y_prob.argmax(axis=-1)

"""

predictions = model.predict(img)
print(np.argmax(predictions))