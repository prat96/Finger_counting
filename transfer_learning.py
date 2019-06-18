# shuf -n 10 -e * | xargs -i mv {} path-to-new-folder
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.applications import MobileNetV2
from keras.models import Model
import numpy as np
import cv2
import os

from keras.regularizers import l2

nbatch = 128

train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=12., width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.15, horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory('./datasets/fingers_new/train/', target_size=(224, 224), color_mode='rgb',
                                              batch_size=nbatch, shuffle=True,
                                              classes=['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                              class_mode='categorical')

test_gen = test_datagen.flow_from_directory('./datasets/fingers_new/test/', target_size=(224, 224), color_mode='rgb',
                                            batch_size=nbatch,
                                            classes=['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                            class_mode='categorical')

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

x = base_model.output
x = Conv2D(64, (3,3), activation='relu', input_shape=(224,224,3))(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu')(x)
x = Conv2D(128, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(256, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
preds = Dense(6, activation='softmax')(x)


"""
x = base_model.output
x = Flatten()(x)
x = Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
x = Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
preds = Dense(6, activation='softmax')(x)
"""

"""
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
"""

model = Model(inputs=base_model.input, outputs=preds)

for i, layer in enumerate(model.layers):
    print(i, layer.name)

for layer in model.layers[:154]:
    layer.trainable = False

for layer in model.layers[154:]:
    layer.trainable = True

model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_TEST = test_gen.n // test_gen.batch_size
print(STEP_SIZE_TEST, STEP_SIZE_TRAIN)

history = model.fit_generator(train_gen, steps_per_epoch=STEP_SIZE_TRAIN, epochs=10, validation_data=test_gen,
                              validation_steps=STEP_SIZE_TEST)


def predict_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    print(predictions.argmax(axis=-1))


predict_img('three_test.jpg')
predict_img('one_test.png')
