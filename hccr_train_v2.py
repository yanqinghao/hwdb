import os
from keras import layers
from keras import models
# from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
# import pickle

base_dir = '../hwdb-dir'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# f = open('../char_dict', 'rb')
# char_dict = pickle.load(f)
# f.close()


def relu():
    return layers.Activation('relu')


def conv_bolck(input_tensor,
               input_filters,
               mid_filters,
               mp=False,
               dropout=None):
    """
    one conv-relu-bn unit
    """
    x = layers.Conv2D(input_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = relu()(x)
    x = layers.Conv2D(mid_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = relu()(x)
    x = layers.Conv2D(input_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = relu()(x)
    if mp:
        x = layers.AveragePooling2D(
            pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    return x


def build_model(img_size, num_classes):
    inputs = layers.Input(shape=(img_size, img_size, 1))
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = relu()(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = relu()(x)
    x = layers.AveragePooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = conv_bolck(x, 96, 64, mp=True)
    x = conv_bolck(x, 128, 96, mp=True)
    x = conv_bolck(x, 256, 192, mp=True)
    x = conv_bolck(x, 488, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=x)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


input_marix = 96
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(input_marix, input_marix),
    batch_size=256,
    class_mode='categorical',
    color_mode='grayscale')
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(input_marix, input_marix),
    batch_size=256,
    class_mode='categorical',
    color_mode='grayscale')
model = build_model(input_marix, 3755)
print(model.summary())
history = model.fit_generator(
    train_generator,
    steps_per_epoch=744835 // 256,
    epochs=300,
    validation_data=validation_generator,
    validation_steps=150200 // 256,
    use_multiprocessing=True,
    workers=10)

model.save('hwdb_kares_gap_300.h5')

res = pd.DataFrame(history.history)
res.to_csv('./result/res_gap_300.csv', index=False)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('./result/ocr_acc_gap_300.png')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
# plt.show()
plt.savefig('./result/ocr_loss.png')
