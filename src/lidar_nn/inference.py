import os
import numpy as np
import tensorflow as tf
from models import dice
import matplotlib.pyplot as plt
import time
from data_generator import DataGenerator

if __name__ == '__main__':
    #load model
    model = tf.keras.models.load_model(os.path.join('models', 'unet.model'), custom_objects={'dice': dice})

    #load image as csv
    img = np.loadtxt("test1.csv", delimiter=",")
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    start = time.time()
    print("start")
    pred = model.predict(img, batch_size=1)
    duration = time.time() - start
    print(duration)
    for i in range(4):
        np.savetxt(os.path.join("pred" + str(i) + ".csv"), pred[0,:,:,i], delimiter = ",")
        plt.imshow(pred[0,:,:,i], cmap='gray')
        plt.waitforbuttonpress()

    #load predictions
    pred0 = np.loadtxt("pred0.csv", delimiter=",")
    pred1 = np.loadtxt("pred1.csv", delimiter=",")
    pred2 = np.loadtxt("pred2.csv", delimiter=",")
    pred3 = np.loadtxt("pred3.csv", delimiter=",")

    preds = np.stack((pred0, pred1, pred2, pred3), axis=-1)
    preds_cleared = np.empty(shape=preds.shape)

    pred = np.full((256, 256), 255, dtype=np.float32)
    objects = ["humans", "dogs", "wall"]

    #show result
    for obj_num, obj_name in enumerate(objects):
        for i in  range(256):
            for j in range(256):
                if preds[i][j][obj_num] > 0.8:
                    preds_cleared[i][j][obj_num] = 1

        plt.imshow(preds_cleared[:,:,obj_num], cmap='gray')
        print(obj_name)
        plt.waitforbuttonpress()