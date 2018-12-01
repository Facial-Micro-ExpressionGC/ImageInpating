from keras.datasets import cifar100
from keras.preprocessing import image

import numpy as np

def load_cifar10():
    (x_train, _), (x_test, _) = cifar100.load_data(label_mode='fine')
    return (x_train, x_test)