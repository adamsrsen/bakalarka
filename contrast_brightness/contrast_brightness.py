import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
import cv2

sample_inp = np.array([cv2.imread("input.jpg") / 255])
sample_out = sample_inp * 2.5 - 1.5
cv2.imwrite(f"output.jpg", sample_out[0] * 255)

inp = Input(shape=(None, None, 3))
out = Conv2D(3, (1, 1), activation='linear')(inp)
model = Model(inputs=inp, outputs=out)

model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae', 'accuracy'])

for i in range(1000):
    model.fit(sample_inp, sample_out, batch_size=1, epochs=5)
    result = model.predict(sample_inp)
    cv2.imwrite(f"learning/epoch_{i}.jpg", result[0] * 255)
    cv2.imwrite(f"learning/epoch_diff_{i}.jpg", np.absolute(sample_out[0] - result[0]) * 255)

test = np.array([cv2.imread("test.jpg") / 255])
model.evaluate(test, test * 2.5 - 1.5)

print(model.layers[1].get_weights())
