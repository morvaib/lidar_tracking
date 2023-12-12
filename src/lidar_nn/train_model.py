import os
import json
import tensorflow as tf
from models import unet, unet_lstm
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import numpy as np

TRAINING_DATA_DIR = ("./training_data")
LASER_DIR ="../laser_coords"
TEST_NAME = "04_03_17_11_sync"

if __name__ == '__main__':
    with open(os.path.join(LASER_DIR, TEST_NAME + ".json"), "r") as json_file:
        laser_data = json.load(json_file)

    with open(os.path.join(TRAINING_DATA_DIR, TEST_NAME + ".json"), "r") as json_file:
        training_data = json.load(json_file)

    train = DataGenerator(laser_data, training_data, batch_size=2, sequence_size=5, sequence_mode=False)
    #test the datagenerator
    x, y = train.__getitem__(2)
    plt.imshow(x[0,], cmap='gray')
    plt.waitforbuttonpress()
    for i in range(4):
        plt.imshow(y[0,:,:,i], cmap='gray')
        plt.waitforbuttonpress()

    model = unet([256, 256, 1])
    #model = unet_lstm([2, 5, 256, 256, 1])
    print(model.summary())
    #tf.keras.utils.plot_model(model, to_file='unet_lstm2.png')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join('models', 'unet_lstm.model'), verbose=1, mode='max',
                            save_best_only=True,  monitor='dice', save_weights_only=False, period=10)

    model.fit_generator(generator=train, steps_per_epoch=len(train), epochs=100, verbose=1, callbacks=checkpoint)