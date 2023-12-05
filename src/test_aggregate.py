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

TESTS = ["04_03_17_11_sync", "03_23_15_16_sync", "05_03_sync"]

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

                    min_dist = 99999
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


if __name__ == "__main__":
    results = []
    for test_name in TESTS:
        input_path = os.path.join(PROJECTED_POINTS_DIR, test_name + '_' + MODEL + ".json")
        with open(input_path, "r") as json_file:
            final_data = json.load(json_file)

        input_path = os.path.join(PROJECTED_CLUSTER_CENTERS_DIR, test_name + '_' + MODEL + ".json")
        with open(input_path, "r") as json_file:
            center_data = json.load(json_file)

        with open(os.path.join(LASER_DIR, test_name + ".json"), "r") as json_file:
            laser_data = json.load(json_file)

        filtered_laser_data = preprocess_laser_data(laser_data)

        t1 = test_avg_dist(final_data, center_points=center_data, laser_data=filtered_laser_data)
        results.extend(t1)

    first = min(results)
    last = 1000
    bins = np.arange(first, last, 40) # fixed bin size

    plt.xlim([first-5, last+5])
    plt.hist(results, bins=bins, alpha=0.5)
    plt.title('A kimenet és a LiDAR adatok közötti eltérés')
    plt.xlabel('Átlagos eltérés [mm]')
    plt.ylabel('Képkocka szám')

    plt.show()
