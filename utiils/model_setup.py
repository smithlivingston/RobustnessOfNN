from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as mplot 


(x_train, y_train), (x_test, y_test) =  tf.keras.datasets.mnist.load_data()
labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

img_rows, img_cols, channels = 28, 28, 1
x_train = x_train / 255
x_test = x_test / 255
num_classes = 10

x_train = x_train.reshape((-1, img_rows, img_cols, channels))
x_test = x_test.reshape((-1, img_rows, img_cols, channels))

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print("Data shapes", x_test.shape, y_test.shape, x_train.shape, y_train.shape)


model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1))) 
model.add(Dense(128, activation='relu')) 
model.add(Dense(num_classes, activation='softmax'))  

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=32, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

image = x_train[0]
plt.imshow(image, cmap='gray')
plt.title(f"Predicted Label:")
plt.axis('off')
plt.show()

input_image = image.reshape(1, 28, 28)
predictions = model.predict(input_image)

predicted_digit = tf.argmax(predictions[0]).numpy()
print("predicted diggit:", predicted_digit)

def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    return signed_grad

image = x_train[0]
image_label = y_train[0]
perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), image_label).numpy()

adversarial = image + perturbations * 0.1

plt.imshow(adversarial.reshape((img_rows, img_cols)), cmap='gray')
plt.title(f"Predicted Label:")
plt.axis('off')
plt.show()

print(labels[model.predict(image.reshape((1, img_rows, img_cols, channels))).argmax()])
print(labels[model.predict(adversarial).argmax()])




