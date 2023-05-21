import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
import matplotlib.pyplot as plt
import cv2
import os

sample_inp = np.array([])
sample_out = np.array([])

for file in os.listdir("input"):
    img = cv2.cvtColor(cv2.imread(f"input/{file}"), cv2.COLOR_BGR2GRAY)
    sample_inp = np.append(sample_inp, img / 255)
    img = cv2.GaussianBlur(img, (3, 3) ,0, 0)
    sobel = np.uint8(np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)) + np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)))[1:-1, 1:-1]
    sample_out = np.append(sample_out, sobel / 255)
    cv2.imwrite(f"output/{file}", sobel)

sample_inp = np.reshape(sample_inp, (len(os.listdir("input")), 479, 361, 1))
sample_out = np.reshape(sample_out, (len(os.listdir("input")), 477, 359))

inp = Input(shape=(None, None, 1))
mid1 = Conv2D(1, (3, 3), activation='relu')(inp)
mid2 = Conv2D(1, (3, 3), activation='relu')(inp)
mid3 = Conv2D(1, (3, 3), activation='relu')(inp)
mid4 = Conv2D(1, (3, 3), activation='relu')(inp)
out = Add()([mid1, mid2, mid3, mid4])
model = Model(inputs=inp, outputs=out)
model.summary()

model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae', 'accuracy'])

history1 = model.fit(sample_inp, sample_out, batch_size=1, epochs=2500)

result = model.predict(sample_inp)

inp = Input(shape=(None, None, 1))
mid1 = Conv2D(1, (3, 3))(inp)
mid2 = Conv2D(1, (3, 3))(inp)
relu1 = relu(mid1)
relu2 = relu(mid2)

out = Add()([relu1, relu1, relu2, relu2, -mid1, -mid2])
model = Model(inputs=inp, outputs=out)
model.summary()

model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae', 'accuracy'])

history2 = model.fit(sample_inp, sample_out, batch_size=1, epochs=2500)

plt.figure(figsize=[10,6])
plt.plot(history1.history['accuracy'],'r',linewidth=2.0)
plt.plot(history2.history['accuracy'],'b',linewidth=2.0)
plt.legend(['Prvý model', 'Druhý model'],fontsize=16)
plt.xlabel('Epochy',fontsize=16)
plt.ylabel('Presnosť',fontsize=16)
plt.title('Porovnanie modelov',fontsize=16)
plt.savefig('training_sobel.png')
