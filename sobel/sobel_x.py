import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
import cv2

img = cv2.imread("input.jpg")
sobel_x = np.uint8(np.absolute(cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)))
cv2.imwrite("output_x.jpg", sobel_x)
sample_inp = np.array([img / 255])
sample_out = np.array([sobel_x / 255])

inp = Input(shape=(None, None, 3))
mid1 = Conv2D(1, (3, 3), activation='relu', padding="same")(inp)
mid2 = Conv2D(1, (3, 3), activation='linear', padding="same")(inp)
out = Add()([mid1, mid1, mid2])
model = Model(inputs=inp, outputs=out)
model.summary()

model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae'])

for i in range(1000):
    model.fit(sample_inp, sample_out, batch_size=1, epochs=5)
    result = model.predict(sample_inp)
    cv2.imwrite(f"learning_x/epoch_{i}.jpg", result[0] * 255)
    cv2.imwrite(f"learning_x/epoch_diff_{i}.jpg", np.absolute(sample_out[0] - np.reshape(result[0], (479, 361))) * 255)

print(model.layers[1].get_weights())
print(model.layers[2].get_weights())