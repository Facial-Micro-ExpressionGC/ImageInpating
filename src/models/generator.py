from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers import (UpSampling2D, LeakyReLU, BatchNormalization, Flatten, Dense,
                          Input, Reshape, Activation)

def build_generator():

    model = Sequential()

    # Encoder
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(32,32,3), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))


    # Fully Connected
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Reshape((4,4,256)))
    
    # Decoder
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation('tanh'))


    masked_img = Input(shape=(32,32,3))
    gen_missing = model(masked_img)

    return Model(masked_img, gen_missing)

