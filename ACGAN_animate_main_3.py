# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 18:04:20 2018
@author: Carl
"""

import os
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
import glob
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from sklearn.utils import shuffle
import time
import cv2
import scipy
import imageio
from PIL import Image
import matplotlib.gridspec as gridspec
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.utils import plot_model
from keras.layers import Conv2D, Conv2DTranspose, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from ACGAN_animate_model_3 import ACGAN
from loadimage import pictureload, return_label_num
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
from scipy.interpolate import spline
K.set_image_dim_ordering('tf')

from collections import deque

np.random.seed(36)

def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 

def load_data(batch_size, image_shape, data = None):
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    all_data_dirlist = pictureload()
    all_data_dirlist = all_data_dirlist.sample(n=batch_size)
    alltable = all_data_dirlist.iloc[:,0]
    # print(all_data_dirlist.iloc[:,1])
    i = 0 
    for index, row in alltable.iteritems():
        image = Image.open(row)
        image = image.resize(image_shape[:-1])
        image = image.convert('RGB') #remove transparent ('A') layer
        image = np.asarray(image)
        image = norm_img(image)
        sample[i,...] = image
        i += 1
    return sample, all_data_dirlist.iloc[:,1]

def save_img_batch(img_batch,img_save_dir):
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(img_batch.shape[0],16,replace=False)
    #print(rand_indices)
    for i in range(16):
        #plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = img_batch[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(img_save_dir,bbox_inches='tight',pad_inches=0)
    # plt.show()   

num_steps = 10000
batch_size = 64
latent_dim = 100
num_classes = return_label_num()
image_shape = (64,64,3)
adam_lr = 0.0002
adam_beta_1 = 0.5
###################### still need modify
img_save_dir = SCRIPT_PATH + "/train_record3"
######################
log_dir = img_save_dir
save_model_dir = img_save_dir


# load discriminator and generator models
acgan = ACGAN(num_classes)
d_model, g_model, gan = acgan.return_model()

g_model.summary()
g_model.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy')
plot_model(g_model, to_file=SCRIPT_PATH+'/model_plots/generate.png')


d_model.summary()
plot_model(d_model, to_file=SCRIPT_PATH+'/model_plots/discriminator.png')
d_model.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
d_model.trainable = False

# build gan model
gan.summary()
gan.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'] )
plot_model(gan, to_file=SCRIPT_PATH+'/model_plots/gan.png')

# loss data record
avg_disc_fake_loss = deque([0], maxlen=250)     
avg_disc_real_loss = deque([0], maxlen=250)
avg_GAN_loss = deque([0], maxlen=250)

for step in range(num_steps): 
    tot_step = step
    print("Begin step: ", tot_step)
    step_begin_time = time.time() 

    # load dataset
    real_data_X, label_batch = load_data(batch_size, image_shape)

    # generate noise to picture
    noise = np.random.normal(0, 0.5, (batch_size, latent_dim))
    sampled_labels = np.random.randint(0, num_classes, batch_size)
    
    fake_data_X = g_model.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)  
    if (tot_step % 10) == 0:
        step_num = str(tot_step).zfill(4)
        save_img_batch(fake_data_X,img_save_dir + "/generateimage/" + step_num + "_image.png")

    d_model.trainable = True
    g_model.trainable = False

    for train_ix in range(3):
        if step % 30 != 0:
            X_real = real_data_X
            # Label Soomthing
            y_real = np.random.uniform(0.7, 1.2, size=(batch_size,))
            aux_y1 = label_batch.values.reshape(-1, )
            dis_metrics_real = d_model.train_on_batch(X_real, [y_real, aux_y1])
            # Label Soomthing
            X_fake = fake_data_X
            y_fake = np.random.uniform(0.0, 0.3, size=(batch_size,))
            aux_y2 = sampled_labels
            # see if the discriminator can figure itself out...
            dis_metrics_fake = d_model.train_on_batch(X_fake, [y_fake, aux_y2])
            print("Disc: real loss: %f fake loss: %f" % (dis_metrics_real[0], dis_metrics_fake[0]))
            avg_disc_fake_loss.append(dis_metrics_fake[0])
            avg_disc_real_loss.append(dis_metrics_real[0])
        else:
            # make the labels the noisy for the discriminator: occasionally flip the labels
            # when training the discriminator
            X_real = fake_data_X
            y_real = np.random.uniform(0.0, 0.3, size=(batch_size,))
            aux_y1 = sampled_labels
            dis_metrics_real = d_model.train_on_batch(X_real, [y_real, aux_y1])
            # Label Soomthing
            X_fake = real_data_X
            y_fake = np.random.uniform(0.7, 1.2, size=(batch_size,))
            aux_y2 = label_batch.values.reshape(-1, )
            # see if the discriminator can figure itself out...
            dis_metrics_fake = d_model.train_on_batch(X_fake, [y_fake, aux_y2])
    
    # train generate
    g_model.trainable = True
    d_model.trainable = False

    noise = np.random.normal(0, 0.5, (batch_size, latent_dim))
    sampled_labels = np.random.randint(0, num_classes, batch_size)
    trick = np.random.uniform(0.7, 1.2, size=(batch_size,))
    gan_metrics = gan.train_on_batch(
        [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels])
    avg_GAN_loss.append(gan_metrics[0])
    print("GAN loss: %f" % (gan_metrics[0]))

    text_file = open(log_dir+"/training_log.txt", "a")
    text_file.write("Step: %d Disc: real loss: %f fake loss: %f GAN loss: %f\n" % (tot_step, avg_disc_real_loss[-1], avg_disc_fake_loss[-1],avg_GAN_loss[-1]))
    text_file.close()
    avg_GAN_loss.append(gan_metrics[0])

    
    end_time = time.time()
    diff_time = int(end_time - step_begin_time)
    print("Step %d completed. Time took: %s secs." % (tot_step, diff_time))
    
    if ((tot_step+1) % 500) == 0:
        print("-----------------------------------------------------------------")
        print("Average Disc_fake loss: %f" % (np.mean(avg_disc_fake_loss)))    
        print("Average Disc_real loss: %f" % (np.mean(avg_disc_real_loss)))    
        print("Average GAN loss: %f" % (np.mean(avg_GAN_loss)))
        print("-----------------------------------------------------------------")
        d_model.trainable = True
        g_model.trainable = True
        g_model.save(save_model_dir+'/models_set/'+str(tot_step)+"_GENERATOR_weights_and_arch.hdf5")
        d_model.save(save_model_dir+'/models_set/'+str(tot_step)+"_DISCRIMINATOR_weights_and_arch.hdf5")


#generator = load_model(save_model_dir+'9999_GENERATOR_weights_and_arch.hdf5')

#generate final sample images
# for i in range(10):
#     noise = np.random.normal(0, 1, size=(batch_size,)+noise_shape)
#     fake_data_X = generator.predict(noise)    
#     save_img_batch(fake_data_X,img_save_dir+"/generateimage/"+"final"+""str(i)+"_image.png")


# """
# #Display Training images sample
# save_img_batch(sample_from_dataset(batch_size, image_shape, data_dir = data_dir),img_save_dir+"_12TRAINimage.png")
# """

# #Generating GIF from PNG
# images = []
# all_data_dirlist = list(glob.glob(img_save_dir+"*_image.png"))
# for filename in all_data_dirlist:
#     img_num = filename.split('\\')[-1][0:-10]
#     if (int(img_num) % 100) == 0:
#         images.append(imageio.imread(filename))
# imageio.mimsave(img_save_dir+'movie.gif', images) 
    
# """
# Alternate way to convert PNG to GIF (ImageMagick):
#     >convert -delay 10 -loop 0 *_image.png animated.gif
# """