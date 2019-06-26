import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import PIL
from PIL import Image
import numpy as np
import time
# import cv2
import argparse

# DEF. PARAMETERS
img_row, img_column = 224, 224
num_channel = 3
num_batch = 1
input_mean = 127.5
input_std = 127.5
floating_model = False

# include the path containing the model (.lite, .tflite)
path_1 = r"./models/mobilenet_v2_1.0_224.tflite"
labels_path = "./models/labels_mobilenet.txt"

"""
parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be classified")
    parser.add_argument("--graph", help=".tflite model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_mean", help="input_mean")
    parser.add_argument("--input_std", help="input standard deviation")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
"""


def load_labels(filename):
    my_labels = []
    input_file = open(filename, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    # print("\nLabels: ", my_labels)
    return my_labels

# TFLITE INTERPRETER CON.
interpreter = tf.lite.Interpreter(path_1)
interpreter.allocate_tensors()
# obtaining the input-output shapes and types
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details, '\n', output_details)
print("\n", input_details[0]['dtype'])
print("\n", input_details[0]['index'])

# INPUT SELECTION
# file selection window for input selection
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
input_img = Image.open(file_path)
# TODO check results after changing BGR2RGB
# input_img = cv2.imread(file_path)
# input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
# input_img = cv2.resize(input_img, (256, 256))

input_img = input_img.resize((img_row, img_column))
input_img = np.expand_dims(input_img, axis=0)

input_img = (np.float32(input_img) - input_mean) / input_std

interpreter.set_tensor(input_details[0]['index'], input_img)

# running inference
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
results = np.squeeze(output_data)

top_k = results.argsort()[-5:][::-1]
labels = load_labels(labels_path)
for i in top_k:
    print('{0:08.6f}'.format(float(results[i] / 255.0)) + ":", labels[i])
