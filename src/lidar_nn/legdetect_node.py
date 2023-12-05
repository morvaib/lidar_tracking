#!/usr/bin/env python
import os
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point
from tracking_humans_and_animals.msg import Legs
import time
import numpy as np
import matplotlib.pyplot as plt
import tf
from scipy.spatial.transform import Rotation
import tensorflow
from models import dice

class LegDetect:
    def __init__(self):
        rospy.Subscriber("/rb1_base/front_laser/scan", LaserScan, self.callback_laser)
        rospy.Subscriber("/rb1_base/map", OccupancyGrid, self.callback_map)
        rospy.Subscriber("/robot_pose", Pose, self.callback_pose)
        self.pub = rospy.Publisher('/legs', Legs)
        self.tf_listener = tf.TransformListener()
        self.pose = []
        self.laser = []
        self.map_plot = []
        self.lasertf_plot_x = []
        self.lasertf_plot_y = []
        self.scale = 1
        self.x_offset = 0
        self.y_offset = 0
        self.model = tensorflow.keras.models.load_model(os.path.join('models', 'unet.model'), custom_objects={'dice': dice})

    def callback_pose(self, msg):
        self.pose = [msg.position.x, msg.position.y, msg.position.z]

    def callback_laser(self, msg):
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges), dtype = np.float32)
        ranges = np.array(msg.ranges, dtype = np.float32)
        self.laser_x = ranges * np.cos(angles)
        self.laser_y = ranges * np.sin(angles)

    def callback_map(self, msg):
        map = np.matrix(np.reshape(msg.data, (msg.info.height, msg.info.width)))
        m = np.empty([msg.info.height, msg.info.width])
        self.map_plot = np.asmatrix(m)
        for i in range(msg.info.height):
            for j in range(msg.info.width):
                self.map_plot[-i,j] = msg.data[i*msg.info.width + j]

    def laser_to_map(self):
        rate = rospy.Rate(5.0)
        while not rospy.is_shutdown():
            try:
                trans, rot = self.tf_listener.lookupTransform('/rb1_base_map', '/rb1_base_front_laser_link', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            r = Rotation.from_quat(rot)
            for i in range(len(self.laser_x)):
                if i == 0:
                    lasertf = r.apply(np.array([self.laser_x[i], self.laser_y[i], 0])) + np.array(trans)
                else:
                    lasertf = np.vstack((lasertf, r.apply(np.array([self.laser_x[i], self.laser_y[i], 0])) + np.array(trans)))
            self.lasertf_plot_x = np.empty(len(self.laser_x))
            self.lasertf_plot_y = np.empty(len(self.laser_y))
            for i in range(len(self.lasertf_plot_x)):
                self.lasertf_plot_x[i] = lasertf[i][0]
                self.lasertf_plot_y[i] = lasertf[i][1]

            pose = r.apply(np.array([0, 0, 0])) + np.array(trans)
            #plt.cla()
            #plt.plot(self.lasertf_plot_x, self.lasertf_plot_y, "ro", markersize=2)
            # img = self.lidar_to_image()
            # img = self.preprocess_img(img)
            # np.savetxt("data.csv", img, delimiter = ",")
            # print("saved")
            #masks = self.detect_legs(img)
            #print(masks.shape)
            # plt.imshow(img, cmap='gray')
            #plt.pause(0.001)
            rate.sleep()

    def lidar_to_image(self):
        img = np.full((256, 256), 255, dtype=np.float32)

        x_min = min(self.lasertf_plot_x)
        x_max = max(self.lasertf_plot_x)
        x_range = abs(x_max  - x_min)
        y_min = min(self.lasertf_plot_y)
        y_max = max(self.lasertf_plot_y)
        y_range = abs(y_max - y_min)
        self.scale = 250/max(x_range, y_range)
        self.x_offset = -1 * x_min
        self.y_offset = -1 * y_min

        for x_point, y_point in zip(self.lasertf_plot_x, self.lasertf_plot_y):
            y = 256 - int((y_point + self.y_offset)*self.scale)
            if y < 256:
                img[y][int((x_point + self.x_offset)*self.scale)] = 0

        return img

    def masks_to_lidar(self, preds):
        leg_msg = Legs()
        objects = ["humans", "dogs", "wall"]
        #preds = np.squeeze(preds, axis=0)
        for obj_num, obj_name in enumerate(objects):
            for i in  range(256):
                for j in range(256):
                    if preds[i][j][obj_num] > 0.8:
                        x = j/self.scale - self.x_offset
                        y = (256 - i)/self.scale - self.y_offset
                        z = 0
                        if objects[obj_num] == "humans":
                            leg_msg.humans.append(Point(x,y,z))
                        elif objects[obj_num] == "dogs":
                            leg_msg.dogs.append(Point(x,y,z))
                        elif objects[obj_num] == "wall":
                            leg_msg.wall.append(Point(x,y,z))

        self.pub.publish(leg_msg)

    def preprocess_img(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def detect_legs(self, img):
        img = self.preprocess_img(img)
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        pred = self.model.predict(img)

        return pred        

if __name__=='__main__':
    rospy.init_node('legdetect_nn')
    leg = LegDetect()

    leg.laser_to_map()

    rospy.spin()