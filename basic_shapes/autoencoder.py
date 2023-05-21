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
            img = cv2.imread(f"input/{category}/{file}", cv2.COLOR_BGR2GRAY) / 255

            images = np.append(images, img)
            labels = np.append(labels, i)

images = np.reshape(images, (len(labels), 28, 28))

training_images, testing_images, training_labels, testing_labels = train_test_split(images, labels, test_size=0.2, shuffle=True)

accuracy = []

for i in range(1,4):
    accuracy.append([])
    for j in range(1,5):
        os.mkdir(f"output/{i}_{j}")

        network_size = pow(2, j)
        latent_size = pow(2, i)
        a = 0
        while a < .9:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(32, 32, 1)),
                tf.keras.layers.Conv2D(network_size * 2, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPool2D((2, 2), padding='same'),
                tf.keras.layers.Conv2D(network_size, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPool2D((2, 2), padding='same'),
                tf.keras.layers.Conv2D(latent_size, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPool2D((4, 4), padding='same'),
                tf.keras.layers.UpSampling2D((4, 4)),
                tf.keras.layers.Conv2DTranspose(latent_size, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2DTranspose(network_size, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2DTranspose(network_size*2, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same'),
            ])

            model.compile(
                optimizer='adam',
                loss="mse",
                metrics=['accuracy']
            )

            model.summary()

            history = model.fit(training_images, training_images, epochs=200)
            a = history.history['accuracy'][-1]
        accuracy[i-1].append(a)

        test_loss, test_acc = model.evaluate(testing_images,  testing_images, verbose=2)

        print('\nTest accuracy:', test_acc)

        result = model.predict(testing_images)
        for k, img in enumerate(result):
            cv2.imwrite(f"output/{i}_{j}/{k}.jpg", img * 255)

print(accuracy)