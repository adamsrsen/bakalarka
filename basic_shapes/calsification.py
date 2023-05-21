from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import cv2
import os

images = np.array([])
labels = np.array([])

for i, category in enumerate(os.listdir("input")):
    if os.path.isdir(f"input/{category}"):
        for file in os.listdir(f"input/{category}"):
            img = cv2.imread(f"input/{category}/{file}", cv2.COLOR_BGR2GRAY)

            images = np.append(images, img)
            labels = np.append(labels, i)

images = np.reshape(images, (len(labels), 32, 32))

training_images, testing_images, training_labels, testing_labels = train_test_split(images, labels, test_size=0.25, shuffle=True, random_state=5)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 1)),
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.MaxPool2D((2, 2), padding='same'),
    tf.keras.layers.Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.MaxPool2D((2, 2), padding='same'),
    tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.MaxPool2D((4, 4), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

model.fit(training_images, training_labels, epochs=500)

test_loss, test_acc = model.evaluate(testing_images,  testing_labels, verbose=2)

print('\nTest accuracy:', test_acc)
