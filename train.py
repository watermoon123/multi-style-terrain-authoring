# -*-coding:utf-8-*-
from __future__ import print_function, division
import scipy

import keras
from keras.datasets import mnist
# from keras_contrib.layers.normalization import InstanceNormalization

from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.losses import *
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import socket
import cv2
from ops import *
import pickle
from functools import partial
from psychopy.visual import filters
from keras_gradient_accumulation import AdamAccumulated
from utils import butter2d

from config import *

from keras.backend.tensorflow_backend import set_session
from tensorboardX import SummaryWriter
from shutil import copyfile


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tf_config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=tf_config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


class MultipleTerrain():
    def __init__(self, weights=None):

        self.terrainFiles = [os.listdir(os.path.join(terrain_png_fd, train_style[i])) for i in range(typeCount)]
        self.fileLen_total = [len(self.terrainFiles[i]) for i in range(typeCount)]

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator_patch()
        def norm_mae(y_true, y_pred):
            return K.mean(K.abs(y_true- y_pred) * (y_true+typeCount / (typeCount - 1)))

        opt = Adam(0.0005, 0.5)
        self.discriminator.compile(loss=[norm_mae for _ in range(scaleCount)],
                                             optimizer=opt
                                             )


        sketch_input = Input(shape=(ter_size, ter_size, 1))

        self.d_global = self.discriminator.outputs[0].shape.as_list()[1]    #32
        self.d_local = self.discriminator.outputs[1].shape.as_list()[1] #128

        self.real_local_dim = 32

        type_input_g = Input(shape=(1, 1, typeCount))
        type_input_l = Input(shape=(self.real_local_dim, self.real_local_dim, typeCount))

        # Build the generator
        self.generator = self.build_generator()


        self.discriminator.trainable = False
        # Discriminators determines validity of translated images / condition pairs
        generate_ter = self.generator([sketch_input, type_input_g, type_input_l])
        valids = self.discriminator([sketch_input, generate_ter])

        self.Get_Gen = K.function([sketch_input, type_input_g, type_input_l], [generate_ter])


        self.combined_soft = Model(inputs=[sketch_input, type_input_g, type_input_l],
                              outputs=valids)

        def soft_mae_g(y_true, y_pred):
            resize_g = K.resize_images(type_input_g, self.d_global, self.d_global, 'channels_last', interpolation='bilinear')
            return K.mean(K.abs(y_true - y_pred) * resize_g)

        def soft_mae_l(y_true, y_pred):
            resize_l = K.resize_images(type_input_l, self.d_local // self.real_local_dim, self.d_local // self.real_local_dim,
                                       'channels_last', interpolation='bilinear')
            return K.mean(K.abs(y_true - y_pred) * resize_l)

        lss = [soft_mae_g, soft_mae_l]
        opt = Adam(0.0005, 0.5)
        self.combined_soft.compile(loss=lss, loss_weights=[2, 1],
                              optimizer=opt)

        self.discriminator.summary()
        self.generator.summary()
        if weights is not None:
            self.generator.load_weights(weights[0])
            self.discriminator.load_weights(weights[1])
        pass

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            if skip_input is not None:
                u = Concatenate()([u, skip_input])
            return u

        type_input_g = Input(shape=(1, 1, typeCount))
        type_input_l = Input(shape=(self.real_local_dim, self.real_local_dim, typeCount))

        # Image input
        sketch_inp = Input(shape=(ter_size, ter_size, 1))
        down_inp = AveragePooling2D()(sketch_inp)

        # Downsampling
        d1 = conv2d(down_inp, self.gf // 4, bn=False)

        d2 = conv2d(d1, self.gf // 2)

        d3 = conv2d(d2, self.gf)
        d3 = Concatenate()([d3, type_input_l])

        d4 = conv2d(d3, self.gf * 2)
        #d4 = Concatenate()([d4, type_input_l])

        d5 = conv2d(d4, self.gf * 4)

        d7 = Concatenate()([d5, UpSampling2D(8)(type_input_g)])
        d7 = conv2d_res(d7, 512, f_size=3, stride=1, in_norm='in_norm', sep=True)
        d7 = conv2d_res(d7, 512, f_size=3, stride=1, in_norm='in_norm', sep=True)
        d7 = conv2d_res(d7, 512, f_size=3, stride=1, in_norm='in_norm', sep=True)
        d7 = conv2d_res(d7, 512, f_size=3, stride=1, in_norm='in_norm', sep=True)

        d7 = Concatenate()([d7, AveragePooling2D(self.real_local_dim // 8)(type_input_l)])
        d7 = conv2d_res(d7, 512, f_size=3, stride=1, in_norm='in_norm', sep=True)
        d7 = conv2d_res(d7, 512, f_size=3, stride=1, in_norm='in_norm', sep=True)
        d7 = conv2d_res(d7, 512, f_size=3, stride=1, in_norm='in_norm', sep=True)
        d7 = conv2d_res(d7, 512, f_size=3, stride=1, in_norm='in_norm', sep=True)

        # Upsampling
        # u1 = deconv2d(d7, d6, self.gf * 8)
        # u2 = deconv2d(u1, d5, self.gf * 8)
        u3 = deconv2d(d7, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf * 4)
        u5 = deconv2d(u4, d2, self.gf * 2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = deconv2d(u6, down_inp, 32)

        u7 = UpSampling2D(size=2)(u7)
        output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model([sketch_inp, type_input_g, type_input_l], output_img)

    def build_discriminator_patch(self):
        sketch_input = Input(shape=(ter_size, ter_size, 1))
        terrain_input = Input(shape=(ter_size, ter_size, 1))

        def d_layer(layer_input, filters, f_size=4, bn=True, stride=2, padding='same'):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=stride, padding=padding)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = InstanceNormalization()(d)
            return d

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([sketch_input, terrain_input])

        combined_imgs = AveragePooling2D()(combined_imgs)

        d1 = d_layer(combined_imgs, self.df, bn=False)
        validities_1 = []
        for _ in range(typeCount):
            d_s = d_layer(d1, self.df * 4, f_size=3)
            validity = Conv2D(1, kernel_size=3, activation='linear', strides=1, padding='same')(d_s)  # linear sigmoid
            validities_1.append(validity)
        validities_1 = Concatenate()(validities_1)

        d2 = d_layer(d1, self.df * 2)

        d3 = d_layer(d2, self.df * 4)
        # d4 = d_layer(d3, self.df * 8)
        # d5 = d_layer(d4, self.df * 8, stride=1)
        validities_3 = []
        for _ in range(typeCount):
            d_s = d_layer(d3, self.df * 4, f_size=3)
            validity = Conv2D(1, kernel_size=3, activation='linear', strides=1, padding='same')(d_s)  # linear sigmoid
            validities_3.append(validity)
        validities_3 = Concatenate()(validities_3)

        return Model([sketch_input, terrain_input], [validities_3, validities_1])

    def train(self, ):

        cur_time = datetime.datetime.now().strftime("%Y-%d-%m-%H-%M-%S")
        cur_log_dir = log_dir + '/' + cur_time
        os.makedirs(cur_log_dir, exist_ok=True)
        weights_dir = cur_log_dir + '/weights'; os.makedirs(weights_dir, exist_ok=True)
        sample_dir = cur_log_dir + '/sampleImgs'; os.makedirs(sample_dir, exist_ok=True)

        copyfile(__file__, cur_log_dir + '/' + os.path.basename(__file__))
        copyfile('ops.py', cur_log_dir + '/' + 'ops.py')
        copyfile('config.py', cur_log_dir + '/' + 'config.py')

        summary_writer = SummaryWriter(log_dir=cur_log_dir)

        start_time = datetime.datetime.now()

        for batch_id in range(iterations*2):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Condition on B and generate a translated version
            elapsed_time = datetime.datetime.now() - start_time
            print_it = 10

            imgs_sketch = np.zeros([batch_size, ter_size, ter_size, 1])
            imgs_ter = np.zeros([batch_size, ter_size, ter_size, 1])
            valids = [-np.ones((batch_size, self.d_global, self.d_global, typeCount)), -np.ones((batch_size, self.d_local, self.d_local, typeCount))]
            type_input_g = np.random.uniform(size=(batch_size, 1, 1, typeCount))
            type_input_g = type_input_g / np.sum(type_input_g, axis=-1, keepdims=True)

            type_input_l = np.random.uniform(size=(batch_size, 1, 1, typeCount))
            type_input_l = type_input_l / np.sum(type_input_l, axis=-1, keepdims=True)
            type_input_l = np.tile(type_input_l, (1, self.real_local_dim, self.real_local_dim, 1))

            for i in range(batch_size):
                img, mask, _type = self.GetFile()
                imgs_ter[i, :, :, :] = img
                imgs_sketch[i, :, :, :] = mask
                for j in range(scaleCount):
                    valids[j][i, :, :, _type] = 1

            genvs = self.Get_Gen([imgs_sketch, type_input_g, type_input_l])
            fake_ter = genvs[0]

            d_loss_real = self.discriminator.train_on_batch([imgs_sketch, imgs_ter], valids)
            d_loss_fake = self.discriminator.train_on_batch([imgs_sketch[:batch_size,:,:,:], fake_ter[:batch_size,:,:,:]],
                    [-np.ones((batch_size, self.d_global, self.d_global, typeCount)),
                     -np.ones((batch_size, self.d_local, self.d_local, typeCount))])


            if batch_id % print_it == 0:
                print(
                    "[Batch %d/%d] [D loss real, D loss fake]" % (
                        batch_id, iterations,)
                    , d_loss_real, d_loss_fake, )
                summary_writer.add_scalar('dis:real:global_loss', d_loss_real[1], batch_id)
                summary_writer.add_scalar('dis:real:local_loss', d_loss_real[2], batch_id)
                summary_writer.add_scalar('dis:fake:global_loss', d_loss_fake[1], batch_id)
                summary_writer.add_scalar('dis:fake:local_loss', d_loss_fake[2], batch_id)
            # -----------------
            #  Train Generator
            # -----------------
            # Train the generators
            for gen_train_it in range(1):  #typeCount
                # soft training
                imgs_sketch = np.zeros([batch_size, ter_size, ter_size, 1])
                imgs_ter = np.zeros([batch_size, ter_size, ter_size, 1])
                valids = [np.ones((batch_size, self.d_global, self.d_global, typeCount)), np.ones((batch_size, self.d_local, self.d_local, typeCount))]

                type_input_g = np.random.uniform(size=(batch_size, 1, 1, typeCount))
                type_input_g = type_input_g / np.sum(type_input_g, axis=-1, keepdims=True)

                type_input_l = np.random.uniform(size=(batch_size, 1, 1, typeCount))
                type_input_l = type_input_l / np.sum(type_input_l, axis=-1, keepdims=True)
                type_input_l = np.tile(type_input_l, (1, self.real_local_dim, self.real_local_dim, 1))
                for i in range(batch_size):
                    img, mask, _type = self.GetFile()
                    imgs_ter[i, :, :, :] = img
                    imgs_sketch[i, :, :, :] = mask
                g_loss = self.combined_soft.train_on_batch([imgs_sketch, type_input_g, type_input_l],
                                                      valids)
                # Plot the progress
                if batch_id % print_it == 0:
                    print(
                        "[Batch %d/%d] "
                        "[G soft loss gan] time: %s\n" % (  # [G loss L1: %f]
                            batch_id, iterations,
                            elapsed_time), g_loss)
                    if gen_train_it == 0:
                        summary_writer.add_scalar('gen:global_loss', g_loss[1], batch_id)
                        summary_writer.add_scalar('gen:local_loss', g_loss[2], batch_id)

            # If at save interval => save generated image samples
            if batch_id % sample_interval == 0:
                test_batch = typeCount      #3
                imgs_sketch = np.zeros([test_batch, ter_size, ter_size, 1])
                imgs_ter = np.zeros([test_batch, ter_size, ter_size, 1])

                for i in range(test_batch):
                    img, mask, _type = self.GetFile()
                    imgs_ter[i, :, :, :] = img
                    imgs_sketch[i, :, :, :] = mask

                r, c = test_batch, typeCount+2
                # fake_A = self.generator.predict(imgs_B)

                fake_ters = []
                for i in range(typeCount):
                    type_input_g = np.zeros((test_batch, 1, 1, typeCount))
                    type_input_l = np.zeros((test_batch, self.real_local_dim, self.real_local_dim, typeCount))
                    type_input_g[:,:,:, i] = 1
                    type_input_l[:,:,:, i] = 1
                    fake_ter = self.Get_Gen([imgs_sketch, type_input_g, type_input_l])[0]
                    fake_ter = postprocess(fake_ter)
                    fake_ters.append(fake_ter)

                gen_imgs = np.concatenate([postprocess(imgs_sketch),
                                           np.concatenate(fake_ters, axis=-1), postprocess(imgs_ter)], axis=-1)

                fig, axs = plt.subplots(r, c)
                for i in range(r):
                    for j in range(c):
                        axs[i, j].imshow(gen_imgs[i, :, :, j])
                        #axs[i, j].set_title(titles[j])
                        axs[i, j].axis('off')
                fig.savefig(sample_dir+"/gen_%d.png" % (batch_id))
                plt.close()

                #type_input_g = np.zeros((typeCount, 1, 1, typeCount))
                #type_input_l = np.zeros((typeCount, self.real_local_dim, self.real_local_dim, typeCount))
                #for i in range(typeCount):
                #    type_input_g[i, :, :, i] = 1    #全局风格
                #    mid_l = self.real_local_dim//2
                #    type_input_l[i, :mid_l, :mid_l, 0] = 1
                #    type_input_l[i, :mid_l, mid_l:, 1] = 1
                #    type_input_l[i, mid_l:, :mid_l, 2] = 1
                #    type_input_l[i, mid_l:, mid_l:, 3] = 1
                #fake_ter = self.Get_Gen([imgs_sketch[:typeCount,...], type_input_g, type_input_l])[0]
                ##fake_ter = np.expand_dims(fake_ter, 1)
                #fake_ter = np.concatenate([fake_ter[0], fake_ter[1], fake_ter[2], fake_ter[3]], axis=0)
                #fake_ter = np.squeeze(postprocess(fake_ter))*255
                #cv2.imwrite(sample_dir+"/gen_%d_map.png" % (batch_id), fake_ter)

            if batch_id >= save_start and batch_id % save_interval == 0:
                self.generator.save_weights(weights_dir + '/generator_tow_scale_stylemap_16_%d.h5' % (batch_id))
                self.discriminator.save_weights(weights_dir + '/discriminator_tow_scale_stylemap_16_%d.h5' % (batch_id))
                pass


    def GetFile(self,):
        terrain_type = np.random.randint(0, typeCount)
        idx = np.random.randint(0, self.fileLen_total[terrain_type])
        r_terrain = cv2.imread(os.path.join(terrain_png_fd, train_style[terrain_type], self.terrainFiles[terrain_type][idx]), -1)

        if r_terrain.dtype == np.uint16:
            r_terrain = r_terrain.astype(np.float)/65535
        elif r_terrain.dtype == np.uint8:
            r_terrain = r_terrain.astype(np.float) / 255
        else:
            raise Exception("unknow image type")

        if np.random.randint(0, 2) == 0:
            r_terrain = cv2.flip(r_terrain, 0)
        if np.random.randint(0, 2) == 0:
            r_terrain = cv2.flip(r_terrain, 1)

        r_terrain = cv2.resize(r_terrain, (ter_size, ter_size))

        # norm
        r_terrain_low_res = butter2d(r_terrain, 0.007, 3)

        return preprocess(r_terrain), preprocess(r_terrain_low_res), terrain_type

def getRandomType():
    ret = np.zeros(shape=(typeCount, ))
    for i in range(typeCount - 1):
        ret[i] = np.random.uniform(0, 1 - np.sum(ret))
    ret[typeCount - 1] = 1 - np.sum(ret)
    assert abs(np.sum(ret) - 1) < 1e-3
    np.random.shuffle(ret)
    return ret

def preprocess(img):
    img = img.astype(np.float)
    img.shape = ter_size, ter_size, 1
    img = (img-0.5)*2
    return img

def postprocess(img):
    img = (img+1)/2
    return img


if __name__ == '__main__':
    gan = MultipleTerrain(weights=None)
    gan.train()