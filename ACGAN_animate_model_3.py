import os
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
import numpy as np
from sklearn.utils import shuffle
import time
import cv2
import tqdm
from PIL import Image
from keras import layers
from keras.layers import Dense, Reshape, Embedding, merge
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten, Dropout
from keras.layers import Input, merge
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras.backend as K
from keras.initializers import RandomNormal
from keras.layers.noise import GaussianNoise
from Minibatch import MinibatchDiscrimination
K.set_image_dim_ordering('tf')

np.random.seed(36)


class ACGAN():
    def __init__(self,NUM_class):

        self.num_classes = NUM_class
        self.latent_dim = 100

        adam_lr = 0.0002
        adam_beta_1 = 0.5
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
    def return_model(self):
        return self.discriminator, self.generator, self.combined
    def build_generator(self):
        
        latent = Input(shape=(self.latent_dim,))
        kernel_init = 'glorot_uniform'
        image_class = Input(shape=(1,), dtype='int32')
        cls_Embedding = Flatten()(Embedding(self.num_classes, self.latent_dim,
                              embeddings_initializer='glorot_normal')(image_class))
        latent_class = layers.multiply([latent, cls_Embedding])

        x = Dense(512*1*1, activation='relu', kernel_initializer='glorot_normal', bias_initializer='Zeros')(latent_class)
        x = Reshape((1,1,512))(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2DTranspose(filters = 256, kernel_size = (4,4), strides = (1,1), padding = "valid", activation='relu'
            , kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = BatchNormalization(momentum = 0.8)(x)

        x = Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = 2, padding = "same", activation='relu'
            , kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = BatchNormalization(momentum = 0.8)(x)

        x = Conv2DTranspose(filters = 64, kernel_size = (4,4), strides = 2, padding = "same", activation='relu'
            , kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = BatchNormalization(momentum = 0.8)(x)

        x = Conv2DTranspose(filters = 32, kernel_size = (4,4), strides = 2, padding = "same", activation='relu'
            , kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = BatchNormalization(momentum = 0.8)(x)

        x = Conv2DTranspose(filters = 3, kernel_size = 5, strides = 2, padding = "same", activation='tanh'
            , kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        return Model([latent, image_class], x)

    def build_discriminator(self):

        dis_inp = Input(shape = (64, 64, 3))
        
        x = Conv2D(filters = 16, kernel_size = (3,3), strides = 2, padding = "same",kernel_initializer='glorot_normal', bias_initializer='Zeros')(dis_inp)
        x = LeakyReLU(0.2)(x)
        # x = Dropout(0.25)(x)

        # 8/26 morning
        x = Conv2D(filters = 32, kernel_size = (3,3), strides = 2, padding = "same",kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        # x = Dropout(0.25)(x)

        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 2, padding = "same",kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        # x = Dropout(0.25)(x)

        x = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = "same",kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        # x = Dropout(0.25)(x)
        
        x = Conv2D(filters = 128, kernel_size = (3,3), strides = 2, padding = "same",kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        # x = Dropout(0.25)(x)

        x = Conv2D(filters = 256, kernel_size = (3,3), strides = 2, padding = "same",kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        # x = Dropout(0.25)(x)

        x = Conv2D(filters = 512, kernel_size = (3,3), strides = 1, padding = "same",kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        # x = Dropout(0.25)(x)
        
        x = Flatten()(x)
          
        fake = Dense(1, activation='sigmoid', name='generation',
                 kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        aux = Dense(self.num_classes, activation='softmax', name='auxiliary',
                kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
    
        return Model(dis_inp, [fake, aux])
