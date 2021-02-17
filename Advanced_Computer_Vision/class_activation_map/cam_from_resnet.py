from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from glob import glob

image_files = glob('./101_ObjectCategories/*/*.jp*g')

height = 224
width = 224
channels = 3

img = image.load_img(np.random.choice(image_files), target_size=(height, width))

resnet = ResNet50(input_shape=(width, height, channels), weights='imagenet', include_top=True)

activation_layer = resnet.get_layer('conv5_block3_out')

model = Model(inputs=resnet.input, outputs=activation_layer.output)

last_layer = resnet.get_layer('predictions')
W = last_layer.get_weights()[0]

x = preprocess_input(np.expand_dims(img, 0))
fmaps = model.predict(x)[0]

prob = resnet.predict(x)
classnames = decode_predictions(prob)[0]
classname = classnames[0][1]
pred = np.argmax(prob)

w = W[:, pred]

cam = fmaps.dot(w)

factor = int(height / np.shape(fmaps)[0])
cam = sp.ndimage.zoom(cam, (factor, factor), order=1)

plt.subplot(121)
plt.imshow(img, alpha=.8)
plt.imshow(cam, cmap='jet',alpha=.6)
plt.subplot(122)
plt.imshow(img)
plt.title(classname)
plt.show()