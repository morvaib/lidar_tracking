import os
import json
import math

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.abspath("./")

PROJECTED_DIR = os.path.join(ROOT_DIR, "projected")
PROJECTED_POINTS_DIR = os.path.join(PROJECTED_DIR, "points")
PROJECTED_CLUSTER_CENTERS_DIR = os.path.join(PROJECTED_DIR, "cluster_centers")
LASER_DIR = os.path.join(ROOT_DIR, "laser_coords")

TEST_NAME = "05_03_sync" # 03_23_15_16_sync, 04_03_17_11_sync, 05_03_sync

MODEL = "det"

# Labor parameters in meter
LAB_X = 5.4
LAB_Y = 6.27

BORDER_TRESHOLD = 7

def check_border(point):
    is_border = False
    if point[0] < BORDER_TRESHOLD or point[1] < BORDER_TRESHOLD or point[0] > LAB_X*100 - BORDER_TRESHOLD or point[1] > LAB_Y*100 - BORDER_TRESHOLD:
        is_border = True

    return is_border

def preprocess_laser_data(laser_data):
    filtered_laser_data = {}
    for frame_id, frame_data in laser_data.items():
        filtered_laser_data[frame_id] = []
        for point in frame_data:
            if not check_border(point):
                filtered_laser_data[frame_id].append(point)

    return filtered_laser_data

def test_avg_dist(final_data, center_points=None, laser_data=None, scale=1000):
    avg_distances = []
    for frame_id, frame_data in final_data.items():
        laser_data_list = list(laser_data.values())
        rate = len(laser_data) / len(final_data)
        ros_frame = math.floor((int(frame_id)-1)*rate)

        min_distances = []
        for center_class, center_map in center_points[frame_id].items():
            for obj_id, center_pt in center_map.items():
                if center_pt is not None:
                    id = obj_id
                    x = center_pt[0]*scale
                    y = center_pt[1]*scale

                    min_dist = 900
                    for point in laser_data_list[ros_frame]:
                        laser_x = point[0]*scale/100
                        laser_y = point[1]*scale/100
                        dist = np.linalg.norm(np.array([laser_x, laser_y]) - np.array([x,y]))
                        if dist < min_dist:
                            min_dist = dist
                    min_distances.append(min_dist)

        if len(min_distances):
            avg_dist = sum(min_distances)/len(min_distances)
            avg_distances.append(avg_dist)

    return avg_distances


def test_velocity(final_data, center_points=None, laser_data=None, scale=1000):
    prev_positions = {}
    all_distances = []
    for frame_id, frame_data in final_data.items():
        distances = []
        for center_class, center_map in center_points[frame_id].items():
            for obj_id, center_pt in center_map.items():
                if center_pt is not None:
                    if obj_id not in prev_positions.keys():
                        prev_positions[obj_id] = None
                    elif prev_positions[obj_id]:
                        x_new = center_pt[0]*scale
                        y_new = center_pt[1]*scale
                        x_prev = prev_positions[obj_id][0]*scale
                        y_prev = prev_positions[obj_id][1]*scale
                        dist = np.linalg.norm(np.array([x_new, y_new]) - np.array([x_prev, y_prev]))
                        distances.append(dist)

                    prev_positions[obj_id] = center_pt

        if len(distances):
            all_distances.append(sum(distances)/len(distances))

    return all_distances


def test_near_points(final_data, center_points=None, laser_data=None, scale=1000):
    nums = []
    for frame_id, frame_data in final_data.items():
        laser_data_list = list(laser_data.values())
        rate = len(laser_data) / len(final_data)
        ros_frame = math.floor((int(frame_id)-1)*rate)

        point_num = 0
        for point in laser_data_list[ros_frame]:
            laser_x = point[0]*scale/100
            laser_y = point[1]*scale/100
            min_dist = 150

            is_close = False
            for center_class, center_map in center_points[frame_id].items():
                for obj_id, center_pt in center_map.items():
                    if center_pt is not None:
                        x = center_pt[0]*scale
                        y = center_pt[1]*scale
                    
                        dist = np.linalg.norm(np.array([laser_x, laser_y]) - np.array([x,y]))
                        if dist < min_dist:
                            min_dist = dist
                            is_close = True

            if is_close:
                point_num +=1

        nums.append(point_num)

    return nums


if __name__ == "__main__":
    input_path = os.path.join(PROJECTED_POINTS_DIR, TEST_NAME + '_' + MODEL + ".json")
    with open(input_path, "r") as json_file:
        final_data = json.load(json_file)

    input_path = os.path.join(PROJECTED_CLUSTER_CENTERS_DIR, TEST_NAME + '_' + MODEL + ".json")
    with open(input_path, "r") as json_file:
        center_data = json.load(json_file)

    with open(os.path.join(LASER_DIR, TEST_NAME + ".json"), "r") as json_file:
        laser_data = json.load(json_file)

    filtered_laser_data = preprocess_laser_data(laser_data)

    t1 = test_avg_dist(final_data, center_points=center_data, laser_data=filtered_laser_data)
    t2 = test_velocity(final_data, center_points=center_data, laser_data=filtered_laser_data)
    t3 = test_near_points(final_data, center_points=center_data, laser_data=filtered_laser_data)

    fig, (ax1,ax2,ax3) = plt.subplots(3)
    fig.suptitle('Eredmények')
    ax1.plot(np.arange(len(t1)), t1)
    ax1.set(xlabel='Képkocka', ylabel='Átlagos eltérés [mm]')
    ax2.plot(np.arange(len(t2)), t2)
    ax2.set(xlabel='Képkocka', ylabel='Átlagos sebesség [mm/frame]')
    ax3.plot(np.arange(len(t3)), t3)
    ax3.set(xlabel='Képkocka', ylabel='Lézer pontok száma [-]')

    plt.show()
    