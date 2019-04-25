from keras import models
from keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import numpy as np

base_dir = '../hwdb-dir'
test_dir = os.path.join(base_dir, 'test')
test_datagen = ImageDataGenerator(rescale=1. / 255)
model = models.load_model('hwdb_kares_gap_300.h5')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(96, 96),
    batch_size=256,
    class_mode='categorical',
    color_mode='grayscale')
merix = model.evaluate_generator(test_generator, workers=10, verbose=1)
print(merix[0], merix[1])
res = model.predict_generator(test_generator, workers=10, verbose=1)
res_label = np.argmax(res, axis=1)
res_df = pd.DataFrame(res_label)
res_df.to_csv('./result/pred.csv')
