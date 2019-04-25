from keras import models
from keras.preprocessing import image
import os
import random
import numpy as np

base_dir = '../hwdb-dir'
test_dir = os.path.join(base_dir, 'test')
label_dir = os.path.join(test_dir, '%05d' % random.randint(0, 3754))
list = os.listdir(label_dir)
f_names = os.path.join(label_dir, random.choice(list))

print(f_names)

images = image.load_img(f_names, grayscale=True, target_size=(96, 96))
x = image.img_to_array(images) / 255.0
x = np.expand_dims(x, axis=0)
model = models.load_model('hwdb_kares.h5')
y = model.predict(x)
res_label = np.argsort(y[0])[::-1][0:5]
print('Predicted:', res_label)
