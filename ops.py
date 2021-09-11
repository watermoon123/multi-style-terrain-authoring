#-*-coding:utf-8-*-

import keras
from keras.layers import *
from keras.layers.merge import _Merge
import tensorflow as tf
#from keras_contrib.layers.normalization import InstanceNormalization
from keras_contrib.layers import InstanceNormalization
import cv2
import matplotlib.pyplot as plt
import math

def conv2d(layer_input, filters, stride=1, f_size=4, in_norm='in_norm'):
    """Layers used during downsampling"""
    d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='same',
               kernel_initializer='he_normal')(layer_input)
    if in_norm=='in_norm':
        d = InstanceNormalization()(d)
    elif in_norm=='bn_norm':
        d = BatchNormalization()(d)
    else:
        pass
    d = LeakyReLU(alpha=0.2)(d)
    return d


def deconv2d(layer_input, skip_input, filters, f_size=4, in_norm='in_norm', up_size=2):
    """Layers used during upsampling"""
    u = Conv2DTranspose(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)

    if in_norm=='in_norm':
        u = InstanceNormalization()(u)
    elif in_norm=='bn_norm':
        u = BatchNormalization()(u)

    u = Activation('relu')(u)
    if not skip_input is None:
        u = Concatenate()([u, skip_input])
    return u

def deconv2d_up(layer_input, skip_input, filters, f_size=4, in_norm='in_norm', up_size=2):
    """Layers used during upsampling"""
    u = UpSampling2D(size=up_size, interpolation='bilinear')(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)

    if in_norm=='in_norm':
        u = InstanceNormalization()(u)
    elif in_norm=='bn_norm':
        u = BatchNormalization()(u)

    u = Activation('relu')(u)
    if not skip_input is None:
        u = Concatenate()([u, skip_input])
    return u

def conv2d_res(layer_input, filters, stride=1, f_size=3, in_norm='in_norm', sep=True):
    """Layers used during downsampling"""
    if sep:
        d = SeparableConv2D(filters, kernel_size=f_size, strides=stride, padding='same',
                   kernel_initializer='TruncatedNormal')(layer_input)
    else:
        d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='same',
                        kernel_initializer='TruncatedNormal')(layer_input)
    if in_norm=='in_norm':
        d = InstanceNormalization()(d)
    elif in_norm=='bn_norm':
        d = BatchNormalization()(d)
    else:
        pass

    if sep:
        shortcut = SeparableConv2D(filters, kernel_size=1, strides=stride, padding='same',
               kernel_initializer='he_normal')(layer_input)
    else:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same',
                               kernel_initializer='he_normal')(layer_input)
    if in_norm=='in_norm':
        shortcut = InstanceNormalization()(shortcut)
    elif in_norm=='bn_norm':
        shortcut = BatchNormalization()(shortcut)
    else:
        pass

    d = Add()([d, shortcut])
    d = LeakyReLU(alpha=0.2)(d)
    return d

def conv2d_res_identity(layer_input, filters, stride=1, f_size=3, in_norm='bn_norm'):
    """Layers used during downsampling"""
    d = SeparableConv2D(filters, kernel_size=f_size, strides=stride, padding='same',
               kernel_initializer='he_normal')(layer_input)

    if in_norm=='in_norm':
        d = InstanceNormalization()(d)
    elif in_norm=='bn_norm':
        d = BatchNormalization()(d)
    d = Add()([layer_input, d])
    d = LeakyReLU(alpha=0.2)(d)
    return d


def deconv2d_res(layer_input, skip_input, filters, f_size=3, in_norm='in_norm'):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = conv2d_res(u, filters=filters, stride=1, f_size = f_size, in_norm = in_norm)
    u = Concatenate()([u, skip_input])
    return u

def conv_block_dense(x, growth_rate, name, norm='in_norm', conv='Separable'):
    bn_axis = 3
    x1 = x
    if norm=='in_norm':
        x1 = InstanceNormalization(name=name + '_0_in')(x1)
    else:
        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x1)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
    if norm == 'bn_norm':
        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    else:
        x1 = InstanceNormalization(name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,padding='same',use_bias=False,name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def dense_block(x, blocks, name, baseFilters=12):
    for i in range(blocks):
        x = conv_block_dense(x, baseFilters, name=name + '_block' + str(i + 1))
    return x

def transition_block(x, reduction, name, norm='in_norm', tpooling='average'):
    bn_axis = 3
    if norm=='in_norm':
        x = InstanceNormalization(name=name + '_bn')(x)
    else:
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    if tpooling=='average':
        x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    else:
        x = MaxPooling2D(2, strides=2, name=name + '_pool')(x)
    return x

def transition_block_up(x, reduction, name, norm='in_norm', upsamp='up'):
    bn_axis = 3
    if norm=='in_norm':
        x = InstanceNormalization(name=name + '_bn')(x)
    else:
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    if upsamp=='up':
        x = UpSampling2D(interpolation='bilinear', name=name + '_up')(x)
    else:
        x = Conv2DTranspose(filters=int(K.int_shape(x)[bn_axis] * reduction), kernel_size=3,
                            padding='same', strides=2, name=name + '_up')(x)
    return x

class LossPlot:
    def __init__(self, savefilename):
        plt.ion()
        self.savefilename = savefilename
        mydpi = 72
        #fig, self.axs = plt.subplots(2, 1,figsize=(920/mydpi, 550/mydpi))
        self.x = []
        self.ls = [[],[],[],[]]
        self.current = 0

    def update(self, loss):
        self.current += 1
        self.x.append(self.current)
        for i in range(4):
            self.ls[i].append(loss[i+1])
        plt.plot(self.x, self.ls[0], color='green', label='Hill')
        plt.plot(self.x, self.ls[1], color='red', label='Mountain')
        plt.plot(self.x, self.ls[2], color='skyblue', label='Plain')
        plt.plot(self.x, self.ls[3], color='blue', label='Frozen earth')
        plt.legend()
        ##
        plt.pause(0.1)
        plt.savefig(self.savefilename)