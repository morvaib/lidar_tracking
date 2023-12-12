import os
import math
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, laser_data, training_data, batch_size, sequence_size, sequence_mode=False):
        self.laser_data = laser_data
        self.training_data = training_data
        self.imshape = (256, 256, 1)
        self.n_classes = 4 #human, dog, wall, background
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.sequence_mode = sequence_mode
        self.sequences = self.genearate_sequences()
        self.on_epoch_end()

    def __len__(self):
        if self.sequence_mode:
            return int(np.floor(len(self.sequences) / self.batch_size))
        return int(np.floor((len(self.training_data) - 5) / self.batch_size))

    def __getitem__(self, index):
        #generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        print(indexes)
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        #updates indexes after each epoch
        if self.sequence_mode:
            self.indexes = np.arange(len(self.sequences))
        else:
            self.indexes = np.arange(5, len(self.training_data))

    def genearate_sequences(self):
        #generate sequences for lstm
        start_num = 5
        stop_num = math.floor((len(self.training_data) - start_num) / self.sequence_size) * self.sequence_size
        return np.arange(start=start_num, stop=stop_num, step=1, dtype=np.float32).reshape(-1, self.sequence_size)

    def lidar_to_image(self, lidar_data_list, frame_num):
        #convert lidar points to image
        img = np.full((256, 256), 0, dtype=np.float32)

        for point in lidar_data_list[frame_num]:
            y = 256 - int(int(point[1])*250/627) - 5
            if y < 256:
                img[y][5 + int(int(point[0])*250/627)] = 255

        return img

    def create_masks(self, img, training_data_frame):
        #create output masks
        channels = []
        walls = np.copy(img)

        for object_name in training_data_frame.keys():
            for object_id in training_data_frame[object_name].keys():
                mask = np.full(shape=img.shape, fill_value=0, dtype=np.float32)
                for point in training_data_frame[object_name][object_id]:
                    mask[256 - int(point[1]*250/627) - 5][int(5 + point[0]*250/627)] = 255

                channels.append(mask)
                walls -= mask
        img -= channels[1]
        img -= channels[3]
        background = 255 - img
        img = 255 - img
        channels = [channels[0], channels[2], walls, background]

        return img,  np.stack(channels, axis=2)/255.0

    def norm_img(self, x):
        #normalize
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def __data_generation(self, indexes):
        #generate training data in batches
        laser_data_list = list(self.laser_data.values())
        rate = len(self.laser_data) / len(self.training_data)

        if self.sequence_mode:
            X = np.empty((self.batch_size, self.sequence_size, self.imshape[0], self.imshape[1], self.imshape[2]), dtype=np.float32)
            Y = np.empty((self.batch_size, self.sequence_size, self.imshape[0], self.imshape[1], self.n_classes),  dtype=np.float32)

            for i in range(len(indexes)):
                frame_nums = np.array(self.sequences[i], dtype=int)
                for seq_num, frame_num in enumerate(frame_nums):
                    ros_frame = math.floor((int(frame_num)-1)*rate)
                    img = self.lidar_to_image(laser_data_list, ros_frame)
                    img, masks = self.create_masks(img, self.training_data[str(frame_num)])
                    X[i, seq_num,] = self.norm_img(np.expand_dims(img, axis=2))
                    Y[i, seq_num,] = masks
                    assert not np.any(np.isnan(X[i, seq_num,]))
                    assert not np.any(np.isnan(Y[i, seq_num,]))
        else:
            X = np.empty((self.batch_size, self.imshape[0], self.imshape[1], self.imshape[2]), dtype=np.float32)
            Y = np.empty((self.batch_size, self.imshape[0], self.imshape[1], self.n_classes),  dtype=np.float32)

            for i, frame_num in enumerate(indexes):
                ros_frame = math.floor((frame_num-1)*rate)
                img = self.lidar_to_image(laser_data_list, ros_frame)
                img, masks = self.create_masks(img, self.training_data[str(frame_num)])
                X[i,] = self.norm_img(np.expand_dims(img, axis=2))
                Y[i,] = masks
                assert not np.any(np.isnan(X[i,]))
                assert not np.any(np.isnan(Y[i,]))


        return X, Y