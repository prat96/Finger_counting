# shuf -n 10 -e * | xargs -i mv {} path-to-new-folder
# ffmpeg -i video.webm thumb%04d.jpg

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
import numpy as np
import cv2
import os
import glob

from keras.utils import plot_model

TRAIN = 1
nbatch = 64
IMG_SIZE = 256


def load_data():
    print("Batch size = ", nbatch, "\n")

    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=12., width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       zoom_range=0.15, shear_range=0.2, horizontal_flip=False)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory('./datasets/fingers_white/train/', target_size=(IMG_SIZE, IMG_SIZE),
                                                  color_mode='rgb',
                                                  batch_size=nbatch, shuffle=True,
                                                  classes=['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                                  class_mode='categorical')

    test_gen = test_datagen.flow_from_directory('./datasets/fingers_white/test/', target_size=(IMG_SIZE, IMG_SIZE),
                                                color_mode='rgb',
                                                batch_size=nbatch,
                                                classes=['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                                class_mode='categorical')

    return train_gen, test_gen


def train_model(train_gen, test_gen):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.6))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
    STEP_SIZE_TEST = test_gen.n // test_gen.batch_size
    print(STEP_SIZE_TEST, STEP_SIZE_TRAIN)

    model.fit_generator(train_gen, steps_per_epoch=STEP_SIZE_TRAIN, epochs=5, validation_data=test_gen,
                        validation_steps=STEP_SIZE_TEST, use_multiprocessing=True, workers=6)

    plot_model(model, to_file='model.png')

    return model, STEP_SIZE_TEST, STEP_SIZE_TRAIN


def predict_img(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    print(img_path, predictions.argmax(axis=-1))


def test_predictions(model):
    os.chdir("./test_images/test_white/")
    for file in glob.glob("*"):
        image_path = file
        predict_img(image_path, model)
    os.chdir("/home/pratheek/Tonbo/Code/finger_counting")


def load_trained_model():
    model_path = './models/fingers_white.h5'
    model = load_model(model_path)

    return model


if __name__ == '__main__':
    if TRAIN == 0:
        model = load_trained_model()
        test_predictions(model)
    else:
        train_gen, test_gen = load_data()
        model, STEP_SIZE_TEST, STEP_SIZE_TRAIN = train_model(train_gen, test_gen)
