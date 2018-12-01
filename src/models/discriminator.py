from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers import (LeakyReLU, BatchNormalization, Flatten, Dense,
                          Input)

def build_discriminator():

    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(16,16,3), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=(16,16,3))
    validity = model(img)

    return Model(img, validity)