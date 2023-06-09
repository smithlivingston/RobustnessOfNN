from keras.engine.sequential import Sequential
from keras.layers.reshaping.flatten import Flatten
from utiils.attacks.fgsm import fgsm, fgsm_attack
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tensorflow.python.keras import layers
from tensorflow.python.keras import models


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the input data
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Define the model architecture
#Build the model object
model = models.Sequential()
# Add the Flatten Layer
model.add(layers.Flatten())
# Build the input and the hidden layers
model.add(layers.Dense(128, activation=tf.nn.relu))
model.add(layers.Dense(128, activation=tf.nn.relu))
# Build the output layer
model.add(layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

# save the trained model
#model.save('mnist.model')

#load the model
model = tf.keras.models.load_model('mnist.model')


predictions = model.predict(x_test)
predicted_labels = tf.argmax(predictions, axis=-1)

# Choose an image to perturb
image_index = 0
input_image = tf.expand_dims(x_test[image_index], axis=0)

# Set the epsilon value
epsilon = 0.1

# Generate the adversarial example using FGSM
perturbed_image = fgsm_attack(model, input_image, epsilon)

# Get the model's predictions for the original and perturbed images
original_pred = model(input_image).numpy().argmax()
perturbed_pred = model(perturbed_image).numpy().argmax()

print("Original Prediction:", original_pred)
print("Perturbed Prediction:", perturbed_pred)



