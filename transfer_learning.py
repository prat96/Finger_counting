# shuf -n 10 -e * | xargs -i mv {} path-to-new-folder

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.applications import MobileNetV2
from keras.models import Model
import numpy as np
from keras import optimizers
import cv2
import os
# from keras import backend as K
# K.set_image_dim_ordering('th')

from keras.regularizers import l2

nbatch = 128
print("Batch size = ", nbatch, "\n")

train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=12., width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.15, shear_range=0.2, horizontal_flip=True, fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory('./datasets/fingers/train/', target_size=(224, 224), color_mode='rgb',
                                              batch_size=nbatch, shuffle=True,
                                              classes=['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                              class_mode='categorical')

test_gen = test_datagen.flow_from_directory('./datasets/fingers/test/', target_size=(224, 224), color_mode='rgb',
                                            batch_size=nbatch,
                                            classes=['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                            class_mode='categorical')

print(test_gen.class_indices, "\n")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

"""
x = base_model.output
x = Conv2D(32, (3, 3), activation='relu')(x)
# x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
# x = MaxPooling2D((2, 2))(x)
# x = Conv2D(128, (3, 3), activation='relu')(x)
# x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
preds = Dense(6, activation='softmax')(x)
"""

x = base_model.output
x = Flatten()(x)
# x = GlobalAveragePooling2D()(x)
x = Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
preds = Dense(6, activation='softmax')(x)

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

adam = optimizers.Adam(lr=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_TEST = test_gen.n // test_gen.batch_size
print(STEP_SIZE_TEST, STEP_SIZE_TRAIN)

model.fit_generator(train_gen, steps_per_epoch=STEP_SIZE_TRAIN, epochs=15, validation_data=test_gen,
                    validation_steps=STEP_SIZE_TEST)


def predict_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    print(predictions.argmax(axis=-1))


predict_img('zero_test.png')
predict_img('one_test.png')
predict_img('two_test.jpg')
predict_img('three_test.jpg')
predict_img('four_test.jpg')
predict_img('five_test.jpg')

