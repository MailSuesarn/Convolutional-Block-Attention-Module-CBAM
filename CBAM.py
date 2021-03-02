"""
Implementation Convolutional Block Attention Module (CBAM) with Residual block (Tensorflow2 & Keras)

This snippet code is a part of the Super AI Engineer article

Author: Suesarn Wilainuch (22p21c0153)

Home: EXP

🔴🟢🔵 let's get started! 🔴🟢🔵
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Reshape, BatchNormalization, Activation, GlobalAveragePooling2D, GlobalMaxPool2D, Concatenate


# ResBlock + CBAM
def conv_block(inputs, filter_num, reduction_ratio, stride=1):
    
    x = inputs
    x = Conv2D(filter_num[0], (1,1), strides=stride, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_num[1], (3,3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_num[2], (1,1), strides=1, padding='same')(x)
    x = BatchNormalization(axis=3)(x)

    # Channel Attention mudule
    avgpool = GlobalAveragePooling2D()(x) # channel avgpool
    maxpool = GlobalMaxPool2D()(x) # channel maxpool
    # Shared MLP
    Dense_layer1 = Dense(filter_num[2]//reduction_ratio, activation='relu') # channel fc1
    Dense_layer2 = Dense(filter_num[2], activation='relu') # channel fc2
    avg_out = Dense_layer2(Dense_layer1(avgpool))
    max_out = Dense_layer2(Dense_layer1(maxpool))

    channel = tf.keras.layers.add([avg_out, max_out])
    channel = Activation('sigmoid')(channel) # channel sigmoid
    channel = Reshape((1,1,filter_num[2]))(channel)
    channel_out = tf.multiply(x, channel)
    
    # Spatial Attention mudule
    avgpool = tf.reduce_mean(channel_out, axis=3, keepdims=True) # spatial avgpool
    maxpool = tf.reduce_max(channel_out, axis=3, keepdims=True) # spatial maxpool
    spatial = Concatenate(axis=3)([avgpool, maxpool])
    # kernel filter 7x7 follow the paper
    spatial = Conv2D(1, (7,7), strides=1, padding='same')(spatial) # spatial conv2d
    spatial_out = Activation('sigmoid')(spatial) # spatial sigmoid

    CBAM_out = tf.multiply(channel_out, spatial_out)

    # residual connection
    r = Conv2D(filter_num[2], (1,1), strides=stride, padding='same')(inputs)
    x = tf.keras.layers.add([CBAM_out, r])
    x = Activation('relu')(x)

    return x