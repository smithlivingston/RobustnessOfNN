import tensorflow as tf

import numpy as np 
import random

import matplotlib.pyplot as mplot 

def adversarial(image, label, model): 
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    return signed_grad


