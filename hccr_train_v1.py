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
    return layers.LeakyReLU(alpha=0.01)


def conv_unit(input_tensor, nb_filters, mp=False, dropout=None):
    """
    one conv-relu-bn unit
    """
    x = layers.ZeroPadding2D()(input_tensor)
    x = layers.Conv2D(nb_filters, (3, 3))(x)
    x = relu()(x)
    x = layers.BatchNormalization(axis=3, momentum=0.66)(x)

    if mp:
        x = layers.MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    return x


def out_block(input_tensor, nb_classes):
    """
    FC output
    """
    x = layers.Flatten()(input_tensor)
    x = layers.Dense(1024)(x)
    x = relu()(x)
    x = layers.BatchNormalization(momentum=0.66)(x)
    x = layers.Dense(256)(x)
    x = relu()(x)
    x = layers.BatchNormalization(momentum=0.66)(x)
    x = layers.Dense(nb_classes)(x)
    x = layers.Activation('softmax')(x)
    return x


def build_model(img_size, num_classes):
    inputs = layers.Input(shape=(img_size, img_size, 1))
    x = layers.ZeroPadding2D()(inputs)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2))(x)
    x = relu()(x)
    x = layers.BatchNormalization(momentum=0.66)(x)
    x = conv_unit(x, 128)
    x = conv_unit(x, 128, mp=True)
    x = conv_unit(x, 256)
    x = conv_unit(x, 256, mp=True)
    x = conv_unit(x, 384)
    x = conv_unit(x, 384)
    x = conv_unit(x, 384, mp=True)
    x = conv_unit(x, 512)
    x = conv_unit(x, 512)
    x = conv_unit(x, 512, mp=True)
    x = out_block(x, num_classes)

    model = models.Model(inputs=inputs, outputs=x)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=256,
    class_mode='categorical',
    color_mode='grayscale')
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=256,
    class_mode='categorical',
    color_mode='grayscale')
model = build_model(64, 3755)
print(model.summary())
history = model.fit_generator(
    train_generator,
    steps_per_epoch=744835 // 256,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=150200 // 256,
    use_multiprocessing=True,
    workers=10)

model.save('hwdb_kares.h5')

res = pd.DataFrame(history.history)
res.to_csv('./result/res.csv', index=False)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('./result/ocr_acc.png')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
# plt.show()
plt.savefig('./result/ocr_loss.png')
