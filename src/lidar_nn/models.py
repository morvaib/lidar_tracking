import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

#dice metrics
def dice(y_true, y_pred, smooth=1.):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

#downblock for unet
def downBlockUnet(inputs, n_filters, kernel, dropout_prob):
    conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel, activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    drop = tf.keras.layers.Dropout(dropout_prob)(conv)
    conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel, activation='elu', kernel_initializer='he_normal', padding='same')(drop)
    bn = tf.keras.layers.BatchNormalization()(conv, training=False)
    pool = tf.keras.layers.MaxPooling2D((2, 2))(bn)
    skip = bn
    return pool, skip

#upblock for unet
def upBlockUnet(up_input, skip_input, n_filters, kernel, dropout_prob):
    up = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(up_input)
    merge = tf.keras.layers.concatenate([up, skip_input])
    conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel, activation='elu', kernel_initializer='he_normal', padding='same')(merge)
    drop = tf.keras.layers.Dropout(dropout_prob)(conv)
    conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel, activation='elu', kernel_initializer='he_normal', padding='same')(drop)
    return conv

#downblock for lstm-unet
def downBlockLSTM(inputs, n_filters, kernel_clstm, kernel_conv):
    print(f"downblock input shape: {np.shape(inputs)}")
    clstm = tf.keras.layers.ConvLSTM2D(filters=n_filters, kernel_size=kernel_clstm, strides=1, stateful=True, \
        return_sequences=True, activation=tf.keras.layers.LeakyReLU(),  padding='same')(inputs)
    orig_shape = clstm.shape
    conv_input = tf.reshape(clstm, [orig_shape[0] * orig_shape[1], orig_shape[2], orig_shape[3], orig_shape[4]])
    conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_conv, use_bias=True, padding='same')(conv_input)
    bn = tf.keras.layers.BatchNormalization(axis=-1)(conv)
    activ = tf.keras.layers.LeakyReLU()(bn)
    conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_conv, use_bias=True, padding='same')(activ)
    bn = tf.keras.layers.BatchNormalization(axis=-1)(conv)
    activ = tf.keras.layers.LeakyReLU()(bn)
    pool = tf.keras.layers.MaxPooling2D((2, 2))(activ)
    out_shape = pool.shape
    activ_down = tf.reshape(pool, [orig_shape[0], orig_shape[1], out_shape[1], out_shape[2], out_shape[3]])
    print(f"downblock output shapes: activ_down: {np.shape(activ_down)}, activ: {np.shape(activ)}")
    return activ_down, activ

#upblock for lstm-unet
def upBlockLSTM(input_sequence, skip_input, n_filters, kernel_conv):
    print(f"upblock input shape: {np.shape(input_sequence)}, skip shape: {np.shape(skip_input)}")
    up = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(input_sequence)
    merge = tf.keras.layers.concatenate([up, skip_input])
    conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_conv, use_bias=True, padding='same')(merge)
    bn = tf.keras.layers.BatchNormalization(axis=-1)(conv)
    activ = tf.keras.layers.LeakyReLU()(bn)

    conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_conv, use_bias=True, padding='same')(activ)
    bn = tf.keras.layers.BatchNormalization(axis=-1)(conv)
    activ = tf.keras.layers.LeakyReLU()(bn)
    print(f"upblock output shape: {np.shape(activ)}")
    return activ

def unet(input_shape):
    i = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    d1, s1 = downBlockUnet(i, 64, 3, 0.1)
    d2, s2 = downBlockUnet(d1, 128, 3, 0.1)
    d3, s3 = downBlockUnet(d2, 256, 3, 0.2)
    d4, s4 = downBlockUnet(d3, 512, 3, 0.2)
    u1 = upBlockUnet(d4, s4, 512, 3, 0.2)
    u2 = upBlockUnet(u1, s3, 256, 3, 0.2)
    u3 = upBlockUnet(u2, s2, 128, 3, 0.1)
    u4 = upBlockUnet(u3, s1, 64, 3, 0.1)
    o = tf.keras.layers.Conv2D(filters=4, kernel_size=1, activation='softmax')(u4)
    print(f"out_shape: {o.shape}")
    model = Model(inputs=i, outputs=o)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=[dice]
    )

    return model

def unet_lstm(input_shape):
    i = tf.keras.layers.Input(batch_size=2, shape=(input_shape[1], input_shape[2], input_shape[3], input_shape[4]))
    d1, s1 = downBlockLSTM(i, 16, 5, 3)
    d2, s2 = downBlockLSTM(d1, 32, 5, 3)
    d3, s3 = downBlockLSTM(d2, 64, 5, 3)
    d4, s4 = downBlockLSTM(d3, 128, 5, 3)
    d4_shape = d4.shape
    d4 = tf.reshape(d4, [d4_shape[0] * d4_shape[1], d4_shape[2], d4_shape[3], d4_shape[4]])
    u1 = upBlockLSTM(d4, s4, 128, 3)
    u2 = upBlockLSTM(u1, s3, 64, 3)
    u3 = upBlockLSTM(u2, s2, 32, 3)
    u4 = upBlockLSTM(u3, s1, 16, 3)
    o = tf.keras.layers.Conv2D(filters=4, kernel_size=1)(u4)
    logits_output_shape = o.shape
    logits_output = tf.reshape(o, [input_shape[0], input_shape[1], logits_output_shape[1],
                                    logits_output_shape[2], logits_output_shape[3]])
    o = tf.keras.layers.Softmax()(logits_output)
    print(f"out_shape: {o.shape}")
    model = Model(inputs=i, outputs=o)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=[dice]
    )

    return model