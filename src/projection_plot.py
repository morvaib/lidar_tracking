import os
import json
import math

import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("./")

PROJECTED_DIR = os.path.join(ROOT_DIR, "datastream/yolo7_deepsort/projected") #detectron2_tracking
PROJECTED_POINTS_DIR = os.path.join(PROJECTED_DIR, "points")
PROJECTED_CLUSTER_CENTERS_DIR = os.path.join(PROJECTED_DIR, "cluster_centers")
LASER_DIR = os.path.join(ROOT_DIR, "laser_coords")

TEST_NAME = "03_23_15_16_sync"
#good: 04_03_17_11_sync
#not so good: 05_08_sync
MODEL = "det"

WITH_LASER = True # LiDAR data
WITH_CENTERS = False # center points

# Labor parameters in meter
LAB_X = 5.4
LAB_Y = 6.27

SLEEP_TIME = 0.01

def plot_data(final_data, center_points=None, laser_data=None, scale=1000):
    fig = plt.figure()
    for frame_id, frame_data in final_data.items():
        plt.cla()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(0, LAB_X*scale)
        plt.ylim(0, LAB_Y*scale)
        plt.text(5000, 5800, frame_id, fontsize=12)
        plt.gca().set_aspect('equal', adjustable='box')
        for object in frame_data:
            x = object['projected_point'][0]
            y = object['projected_point'][1]
            plt.plot(x*scale, y*scale, 'bo')
            #plt.text(x*scale, y*scale, object['class'])
            #plt.text(x*scale, y*scale + scale/5, "id:" + str(object['id']))
            #plt.text(x*scale, y*scale + scale/5, "cam:" + str(object['cam_id']))

        if center_points is not None:
            for center_class, center_map in center_points[frame_id].items():
                for obj_id, center_pt in center_map.items():
                    if center_pt is not None:
                        id = obj_id
                        x = center_pt[0]
                        y = center_pt[1]
                        plt.plot(x*scale, y*scale, 'co')
                        plt.text(x*scale, y*scale, center_class)
                        plt.text(x*scale, y*scale + scale/5, "id:" + str(id))

        if laser_data is not None:
            laser_data_list = list(laser_data.values())
            rate = len(laser_data) / len(final_data)
            ros_frame = math.floor((int(frame_id)-1)*rate)
            laser_x = []
            laser_y = []

            for point in laser_data_list[ros_frame]:
                laser_x.append(point[0]*scale/100)
                laser_y.append(point[1]*scale/100)
            plt.plot(laser_x, laser_y, 'ro', markersize=2)

        plt.pause(SLEEP_TIME)
        plt.waitforbuttonpress()

if __name__ == "__main__":
    if MODEL is not None:
        input_path = os.path.join(PROJECTED_POINTS_DIR, TEST_NAME + '_' + MODEL + ".json")
    else:
        input_path = os.path.join(PROJECTED_POINTS_DIR, TEST_NAME + ".json")

    with open(input_path, "r") as json_file:
        final_data = json.load(json_file)

    if WITH_CENTERS:
        if MODEL is not None:
            input_path = os.path.join(PROJECTED_CLUSTER_CENTERS_DIR, TEST_NAME + '_' + MODEL + ".json")
        else:
            input_path = os.path.join(PROJECTED_CLUSTER_CENTERS_DIR, TEST_NAME + ".json")

        with open(input_path, "r") as json_file:
            center_data = json.load(json_file)

    if WITH_LASER:
        with open(os.path.join(LASER_DIR, TEST_NAME + ".json"), "r") as json_file:
            laser_data = json.load(json_file)

    if WITH_LASER and WITH_CENTERS:
        plot_data(final_data, center_points=center_data, laser_data=laser_data)
    elif WITH_CENTERS:
        plot_data(final_data, center_points=center_data)
    elif WITH_LASER:
        plot_data(final_data, laser_data=laser_data)
    else:
        plot_data(final_data)