from keras.models import Model, Sequential
import tensorflow as tf 


def fgsm(model, clean_image, epsilon, label):
    modelPrediction = model(clean_image)
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    targetLabel = tf.one_hot(label, modelPrediction.shape[-1])
    targetLabel =  tf.reshape(targetLabel, (1, modelPrediction.shape[-1]))

    # tensorImage =  tf.convert_to_tensor(clean_image)

    with tf.GradientTape() as tape:
        tape.watch(clean_image)
        prediction = model(clean_image)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(prediction, tf.zeros_like(prediction))

    gradient = tape.gradient(loss, clean_image)
    signedgrad = tf.sign(gradient)
    perturbed_image  = clean_image + epsilon * signedgrad
    perturbed_image  = tf.clip_by_value(perturbed_image, 0, 1)  # Clip values to [0, 1]
    return perturbed_image
    # adversarialImage = clean_image + epsilon*perturbation
    #untargeted
    #  adversarial_image = clean_image - epsilon*perturbation


def fgsm_attack(model, input_image, epsilon):

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        predicted_labels = tf.argmax(prediction, axis=-1)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(predicted_labels, tf.zeros_like(prediction))
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    perturbed_image = input_image + epsilon * signed_grad
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)  # Clip values to [0, 1]
    return perturbed_image


