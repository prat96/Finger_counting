from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from PIL import Image

from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper


def load_labels(filename):
    my_labels = []
    input_file = open(filename, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels


if __name__ == "__main__":
    file_name = "/tmp/grace_hopper.bmp"
    model_file = "/tmp/mobilenet_v1_1.0_224_quant.tflite"
    label_file = "/tmp/labels.txt"
    input_mean = 127.5
    input_std = 127.5
    floating_model = False

    interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    if input_details[0]['dtype'] == type(np.float32(1.0)):
        floating_model = True

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(file_name)
    img = img.resize((width, height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        if floating_model:
            print('{0:08.6f}'.format(float(results[i])) + ":", labels[i])
        else:
            print('{0:08.6f}'.format(float(results[i] / 255.0)) + ":", labels[i])
